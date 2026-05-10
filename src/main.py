import yaml
from data_utils import HateSpeechDataset
from definitions import HateSpeechDefinition
from model_utils import load_model
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional
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


def predict(
    dataset: datasets.Dataset,
    model,
    tokenizer,
    definition: Optional[HateSpeechDefinition] = None,
    prompting: Optional[Prompting] = None,
    batch_size: int = 8,
    num_batches: int = 1e9,
    generation_config: Optional[dict] = None
):
    generation_config = generation_config or {}
    max_new_tokens = generation_config.get("max_new_tokens", 100)
    do_sample = generation_config.get("do_sample", False)
    temperature = generation_config.get("temperature", 1.0)
    top_p = generation_config.get("top_p", 1.0)

    texts = []
    predictions = []
    confidence_scores = []
    reasons = []
    labels = []
    problematic_generations = []
    iterator = list(dataset)
    if num_batches == 1e9:
        num_samples = len(iterator)
    else:
        num_samples = num_batches * batch_size
    for start in tqdm(range(0, num_samples, batch_size), desc="Predicting..."):
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
            # t0 = time.time()
            outputs = model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                **generation_kwargs,
            )
            # t1 = time.time()
            # print(f"Time taken: {t1 - t0:.3f} seconds for batch generation")
            # raise Exception("Stop here")

        sequences = outputs.sequences
        logits = outputs.scores
        prompt_len = int(inputs["input_ids"].shape[1])
        for j in range(len(batch)):
            i = start + j
            new_tokens = sequences[j][prompt_len:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prediction_index = answer.find("PREDICTION:")
            answer = answer[prediction_index:].strip()

            lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
            if len(lines) < 2:
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue

            prediction_text = lines[0]
            if not prediction_text.lower().startswith("prediction:"):
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue

            prediction = prediction_text.split(":", 1)[1].strip().lower()
            if prediction in ["non-hateful", "non hateful", "not hate speech", "not hate-speech"]:
                prediction = "non-hateful"
            elif prediction in ["hateful", "hate speech", "hate-speech"]:
                prediction = "hateful"
            else:
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue


            reason_text = lines[1]
            if not reason_text.lower().startswith("reason:"):
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue
            reason = reason_text.split(":", 1)[1].strip()

            # TODO: Calculate confidence score
            # confidence_score = torch.max(logits[:][j],dim=1)
            predictions.append(prediction)
            # confidence_scores.append(confidence_score)
            reasons.append(reason)
            texts.append(batch_texts[j])
            labels.append(batch_labels[j])

    return pd.DataFrame(
        {
            "text": texts,
            "prediction": predictions,
            # "confidence_score": confidence_scores,
            "reason": reasons,
            "label": labels,
        }
    ), problematic_generations

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

    datasets = []
    for dataset_name in tqdm(config["datasets"], "Loading datasets"):
        if config["datasets"][dataset_name]["enable"]:
            definitions = []
            for hate_speech_definition in config["datasets"][dataset_name]["hate_speech_definitions"]:
                definition = HateSpeechDefinition.from_config(hate_speech_definition)
                definitions.append(definition)
            datasets.append(HateSpeechDataset.load_dataset(dataset_name, config["datasets"][dataset_name]["path"], config["datasets"][dataset_name]["text_column"], config["datasets"][dataset_name]["label_column"], definitions))
    
    print("Datasets loaded")
    print(*[f"{dataset.name}: {dataset.dataset.num_rows}" for dataset in datasets])

    models = {}
    for model_name in tqdm(config["models"], "Loading models"):
        if config["models"][model_name]["enable"]:
            models[model_name] = load_model(config["models"][model_name]["path"], DEVICE)
    
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
    for definition in config["extra_hate_speech_definitions"]:
        definition = HateSpeechDefinition.from_config(definition)
        extra_definitions.append(definition)
    

    print("-" * 100)
    dataset_names = []
    model_names = []
    prompting_names = []
    definition_names = []
    macro_f1_scores = []
    total_num_experiments = len(datasets) * len(models) * len(prompting) * sum([len(dataset.hate_speech_definitions) for dataset in datasets]) + len(extra_definitions)
    num_experiments_completed = 0
    for dataset in datasets:
        for model in models:
            current_model, current_tokenizer = models[model]
            model_name = getattr(current_model, "name_or_path", "model").replace("/", "_")
            for prompting_method in prompting:
                for definition in dataset.hate_speech_definitions + extra_definitions:
                    experiment_folder = os.path.join(run_folder, dataset.name, model_name, definition.name, prompting_method.name)
                    os.makedirs(experiment_folder)
                    print(f"Running experiment {num_experiments_completed}/{total_num_experiments}:\nmodel: {model_name}\ndataset: {dataset.name}\ndefinition: {definition.name}\nprompting: {prompting_method.name}")
                    print("-" * 100)
                    predictions_df, problematic_generations = predict(
                        dataset,
                        current_model,
                        current_tokenizer,
                        definition,
                        prompting_method,
                        num_batches=1 if debug_mode else 1e9,
                        generation_config=generation_config,
                    )
                    print("A total of " + str(len(problematic_generations)) + " problematic generations were found")
                    print(problematic_generations)
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
                    with open(os.path.join(experiment_folder, "classification_report.txt"), "w") as f:
                        f.write(text_report)
                    with open(os.path.join(experiment_folder, "problematic_generations_indices.txt"), "w") as f:
                        f.write(str(problematic_generations))
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
        }
    ).to_csv(os.path.join(run_folder, "macro_f1_scores.csv"), index=False)



