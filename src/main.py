import yaml
from data_utils import HateSpeechDataset
from definitions import Domain, HateSpeechDefinition
from model_utils import load_model
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional, Union

import datasets
from sklearn.metrics import classification_report
import torch
import pandas as pd
import time
import os
import shutil
import random
import numpy as np
from prompting import ZeroShotPrompting, FewShotPrompting, ChainOfThoughtPrompting, Prompting
load_dotenv()

def read_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# def _build_prompt(text: str, definition: Optional[HateSpeechDefinition]) -> str:
#     definition_block = ""
#     if definition:
#         definition_block = "HATE SPEECH DEFINITION:\n" + definition.prompt_text() + "\n"

#     prompt = f"""Classify the following TEXT as hate speech or not hate speech{" based on HATE SPEECH DEFINITION" if definition else ""}.
#     Your prediction should be in the following format:
#         PREDICTION: either 'hateful' or 'non-hateful'
#         CONFIDENCE SCORE: a number between 0 and 1 which shows the confidence in your answer
#         REASON: only one line explanation for your prediction and confidence score
        
#     {definition_block}
    
#     TEXT: {text}
    
#     PREDICTION:"""
#     return prompt

def calculate_confidence_score(
    step_logits: torch.Tensor,
    chosen_token_ids: torch.Tensor,
    excluded_token_ids: Optional[torch.Tensor] = None,
) -> float:
    """
    Geometric mean of p(chosen_token_t | context) over decoding steps.

    Steps where the emitted token id is in ``excluded_token_ids`` (typically
    ``tokenizer.all_special_ids``: pad, eos, bos, …) are omitted so the score
    reflects only “content” tokens (e.g. the label string), not stop/pad
    machinery. With left-padded prompts, prompt padding is not in this slice
    anyway; this mainly drops trailing EOS and any rare special emissions.
    """
    if step_logits.ndim != 2:
        raise ValueError("step_logits must be (T, vocab_size).")
    t_steps = step_logits.shape[0]
    if t_steps == 0:
        return float("nan")
    t_use = min(t_steps, int(chosen_token_ids.shape[0]))
    if t_use == 0:
        return float("nan")
    logits = step_logits[:t_use].float()
    chosen = chosen_token_ids[:t_use].long()
    log_probs = torch.log_softmax(logits, dim=-1)
    idx = torch.arange(t_use, device=log_probs.device)
    token_logp = log_probs[idx, chosen]

    valid = torch.ones(t_use, dtype=torch.bool, device=log_probs.device)
    if excluded_token_ids is not None and excluded_token_ids.numel() > 0:
        ex = excluded_token_ids.to(chosen.device).long()
        valid &= ~torch.isin(chosen, ex)

    if not bool(valid.any()):
        return float("nan")
    token_logp = token_logp[valid]
    return float(torch.exp(token_logp.mean()).item())


def predict(
    dataset: datasets.Dataset,
    model,
    tokenizer,
    definition: Optional[HateSpeechDefinition] = None,
    prompting: Optional[Prompting] = None,
    batch_size: int = 8,
    num_batches: int = 1e9,
    generation_config: Optional[dict] = None,
    num_generation_retries: int = 5,
    retry_number: int = 1,

):
    generation_config = generation_config or {}
    max_new_tokens = generation_config.get("max_new_tokens", 100)
    do_sample = generation_config.get("do_sample", False)
    temperature = generation_config.get("temperature", 1.0)
    top_p = generation_config.get("top_p", 1.0)

    texts = []
    predictions = []
    confidence_scores = []
    labels = []
    problematic_generations = []
    iterator = list(dataset)
    if num_batches == 1e9:
        num_samples = len(iterator)
    else:
        num_samples = num_batches * batch_size
    for start in tqdm(range(0, num_samples, batch_size), desc=f"Predicting (Attempt {retry_number} of {num_generation_retries})..."):
        batch = iterator[start : start + batch_size]
        system_prompt = prompting.build_system_prompt(definition)
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["text"]},
            ]
            for item in batch
        ]
        batch_labels = [item["label"] for item in batch]
        batch_texts = [item["text"] for item in batch]
        
        prompt_strings = [
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conv in conversations
        ]
        inputs = tokenizer(
            prompt_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            # if DEVICE.type == "cuda":
            #     generation_kwargs["generator"] = torch.Generator(device="cuda").manual_seed(generation_config.get("seed") + start)
            # elif DEVICE.type == "cpu":
            #     generation_kwargs["generator"] = torch.Generator().manual_seed(generation_config.get("seed") + start)

        with torch.no_grad():
            outputs = model.generate(
                output_scores=True,
                return_dict_in_generate=True,
                **inputs,
                **generation_kwargs,
            )

        sequences = outputs.sequences
        scores = torch.stack(outputs.scores, dim=1)

        special_ids = torch.tensor(
            sorted({int(t) for t in getattr(tokenizer, "all_special_ids", []) if t is not None}),
            dtype=torch.long,
            device=scores.device,
        )

        prompt_len = int(inputs["input_ids"].shape[1])
        for j in range(len(batch)):
            i = start + j
            new_tokens = sequences[j][prompt_len:]
            line_break_encoded = torch.tensor(tokenizer.encode("\n", add_special_tokens=False))
            line_break_index = 0
            for k in range(len(new_tokens)-1, 0, -1):
                found = True
                for l in range(len(line_break_encoded)):
                    if new_tokens[k-l] != line_break_encoded[l]:
                        found = False
                        break
                if found:
                    line_break_index = k
                    break

            _meta = tokenizer.decode(new_tokens[:line_break_index], skip_special_tokens=True).strip().lower()
            prediction = tokenizer.decode(new_tokens[line_break_index:], skip_special_tokens=True).strip().lower()
            answer = None
            if prediction == "non-hateful" or prediction.startswith("non-hateful"):
                answer = "non-hateful"
            elif prediction == "hateful" or prediction.startswith("hateful"):
                answer = "hateful"
            elif prediction in ("non hateful", "not hate speech", "not hate-speech"):
                answer = "non-hateful"
            elif prediction in ("hate speech", "hate-speech"):
                answer = "hateful"

            if answer is None:
                problematic_generations.append({"index": i, "text": batch_texts[j], "answer": prediction, "label": batch_labels[j]})
                continue

            chosen_ids = sequences[j, prompt_len + line_break_index:]
            confidence_score = calculate_confidence_score(
                scores[j],
                chosen_ids,
                excluded_token_ids=special_ids if special_ids.numel() else None,
            )
            predictions.append(answer)
            confidence_scores.append(confidence_score)
            texts.append(batch_texts[j])
            labels.append(batch_labels[j])

    predictions_df = pd.DataFrame(
        {
            "text": texts,
            "prediction": predictions,
            "confidence_score": confidence_scores,
            "label": labels,
            "retry_number": retry_number,
        }
    )
    problematic_generations_df = pd.DataFrame(problematic_generations)
    if len(problematic_generations) > 0 and retry_number < num_generation_retries:
        predictions_df2, problematic_generations_df2 = predict(
            datasets.Dataset.from_pandas(problematic_generations_df),
            model,
            tokenizer,
            definition,
            prompting,
            batch_size,
            num_batches,
            generation_config,
            num_generation_retries,
            retry_number + 1,
        )
        predictions_df = pd.concat([predictions_df, predictions_df2])
        problematic_generations_df = problematic_generations_df2
    return predictions_df, problematic_generations_df

if __name__ == "__main__":
    run_id = time.strftime("%m_%d_%H:%M:%S")
    run_folder = os.path.join('runs', run_id)
    os.makedirs(run_folder)
    print("Run ID: " + run_id)

    print("CUDA available: " + str(torch.cuda.is_available()))
    print("MPS available: " + str(torch.backends.mps.is_available()))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: " + str(DEVICE)
    )
    config = read_config("config.yaml")
    shutil.copy("config.yaml", os.path.join(run_folder, "config.yaml"))

    debug_mode = config["debug_mode"]
    print("Debug mode: " + str(debug_mode))

    generation_config = config.get("generation", {})
    seed = generation_config.get("seed")
    seed_everything(seed)
    print(f"Seed set to: {seed}")
    print(f"Generation config: {generation_config}")

    domain = None
    domain_path = config.get("hate_speech_criteria_domain")
    if domain_path:
        try:
            domain = Domain.load(domain_path)
            print(f"Domain loaded from {domain_path} with {sum(1 for _ in domain.iter_leaves())} leaf aspects.")
        except Exception as e:
            print(f"Error loading domain from {domain_path}: {e}")

    datasets_list = []
    for dataset_name in tqdm(config["datasets"], "Loading datasets"):
        if config["datasets"][dataset_name]["enable"]:
            definitions = []
            for hate_speech_definition in config["datasets"][dataset_name]["hate_speech_definitions"]:
                try:
                    definition = HateSpeechDefinition.load_definition(hate_speech_definition, domain=domain)
                except Exception as e:
                    print(f"Error loading definition {hate_speech_definition.get('type')}: {e}")
                    continue
                definitions.append(definition)
            try:
                # datasets_list.append(HateSpeechDataset("problematic", datasets.load_dataset("csv", data_files="data/problematic_generations.csv"), definitions))
                datasets_list.append(HateSpeechDataset.load_dataset(dataset_name, config["datasets"][dataset_name]["path"], config["datasets"][dataset_name]["text_column"], config["datasets"][dataset_name]["label_column"], definitions))
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue
    print("Datasets loaded")
    print(*[f"{dataset.name}: {dataset.dataset.num_rows}" for dataset in datasets_list])

    models = {}
    for model_name in tqdm(config["models"], "Loading models"):
        if config["models"][model_name]["enable"]:
            try:
                models[model_name] = load_model(config["models"][model_name]["path"], DEVICE)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                continue
    
    print("Models loaded")
    print(*[f"{name}: {model[0]}" for name, model in models.items()])

    prompting = []
    for prompting_config in config["prompting"]:
        if prompting_config["type"] == "zero-shot":
            prompting.append(ZeroShotPrompting())
        elif prompting_config["type"] == "few-shot":
            prompting.append(FewShotPrompting(num_shots=prompting_config["num_shots"], few_shot_mode=prompting_config["few_shot_mode"]))
        elif prompting_config["type"] == "chain-of-thought":
            prompting.append(ChainOfThoughtPrompting())
        else:
            raise ValueError(f"Invalid prompting type: {prompting_config['type']}")

    print("Prompting methods loaded")
    print(*[f"{prompting_method.name}" for prompting_method in prompting])


    extra_definitions = []
    for definition in config.get("extra_hate_speech_definitions") or []:
        try:
            extra_definitions.append(HateSpeechDefinition.load_definition(definition, domain=domain))
        except Exception as e:
            print(f"Error loading extra definition {definition.get('type')}: {e}")
    

    print("-" * 100)
    dataset_names = []
    model_names = []
    prompting_names = []
    definition_names = []
    macro_f1_scores = []
    percentage_of_problematic_generations = []
    total_num_experiments = len(datasets_list) * len(models) * len(prompting) * sum([len(dataset.hate_speech_definitions) for dataset in datasets_list]) + len(extra_definitions)
    print(f"Total number of experiments: {total_num_experiments}\n")
    num_experiments_completed = 0
    for dataset in datasets_list:
        for model in models:
            current_model, current_tokenizer = models[model]
            model_name = getattr(current_model, "name_or_path", "model").replace("/", "_")
            for prompting_method in prompting:
                for definition in dataset.hate_speech_definitions + extra_definitions:
                    experiment_folder = os.path.join(run_folder, dataset.name, model_name, definition.name, prompting_method.name)
                    os.makedirs(experiment_folder) 
                    print("-" * 100)
                    print(f"Running experiment {num_experiments_completed}/{total_num_experiments}:\nmodel: {model_name}\ndataset: {dataset.name}\ndefinition: {definition.name}\nprompting: {prompting_method.name}\n")
                   
                    predictions_df, problematic_generations_df = predict(
                        dataset,
                        current_model,
                        current_tokenizer,
                        definition,
                        prompting_method,
                        num_batches=1 if debug_mode else 1e9,
                        generation_config=generation_config,
                        num_generation_retries=config.get("num_generation_retries", 5),
                    )
                    print("A total of " + str(len(problematic_generations_df)) + " problematic generations were found")
                    print(problematic_generations_df)
                    text_report = classification_report(
                        predictions_df["label"],
                        predictions_df["prediction"],
                        labels=["non-hateful", "hateful"],
                        target_names=["non-hateful", "hateful"],
                        zero_division=1,
                    )
                    print(text_report)
                    dict_report = classification_report(
                        predictions_df["label"],
                        predictions_df["prediction"],
                        labels=["non-hateful", "hateful"],
                        target_names=["non-hateful", "hateful"],
                        zero_division=1,
                        output_dict=True,
                    )
                    macro_f1_score = dict_report["macro avg"]["f1-score"]
                    dataset_names.append(dataset.name)
                    model_names.append(model_name)
                    definition_names.append(definition.name)
                    prompting_names.append(prompting_method.name)
                    macro_f1_scores.append(macro_f1_score)
                    percentage_of_problematic_generations.append(len(problematic_generations_df) / len(predictions_df))
                    with open(os.path.join(experiment_folder, "classification_report.txt"), "w") as f:
                        f.write(text_report)
                    problematic_generations_df.to_csv(os.path.join(experiment_folder, "problematic_generations.csv"), index=False)
                    with open(os.path.join(experiment_folder, "prompt.txt"), "w") as f:
                        f.write(prompting_method.build_system_prompt(definition))
                    if config["save_predictions"]:
                        predictions_df.to_csv(os.path.join(experiment_folder, "predictions.csv"), index=False)
                    num_experiments_completed += 1

    pd.DataFrame(
        {
            "dataset": dataset_names,
            "model": model_names,
            "prompting": prompting_names,
            "definition": definition_names,
            "macro_f1_score": macro_f1_scores,
            "percentage_of_problematic_generations": percentage_of_problematic_generations,
        }
    ).to_csv(os.path.join(run_folder, "scores_comparison.csv"), index=False)



