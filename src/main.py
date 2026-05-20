import yaml
from data_utils import HateSpeechDataset
from definitions import HateSpeechDefinition
from model_utils import load_model
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Literal, Optional, Sequence

import datasets
from sklearn.metrics import classification_report, confusion_matrix
import torch
import pandas as pd
import time
import os
import shutil
import random
import numpy as np
import gc
from prompting import ZeroShotPrompting, FewShotPrompting
from chat_utils import build_prompt
from model_utils import load_embedding_model
import traceback
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

def calculate_confidence_score(
    chosen_logits: torch.Tensor,
    chosen_ids: torch.Tensor,
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
    
    # Encoder–decoder models (e.g. T5) prepend a decoder-start token (often pad id 0)
    # without a corresponding entry in ``outputs.scores``, so ``len(token_ids)`` can
    # be one greater than ``step_logits.shape[0]``. 
    n_tok = int(chosen_ids.shape[0])
    n_step = int(chosen_logits.shape[0])
    if n_tok == n_step + 1:
        chosen_ids=chosen_ids[1:]
    if n_tok > n_step + 1:
        chosen_ids=chosen_ids[-n_step:]
    else:
        chosen_logits=chosen_logits[-n_tok:]
    if chosen_ids.shape[0] == 0:
        return float("nan")

    log_probs = torch.log_softmax(chosen_logits, dim=-1)

    valid = torch.ones(chosen_ids.shape[0], dtype=torch.bool, device=log_probs.device)
    if excluded_token_ids is not None and excluded_token_ids.numel() > 0:
        ex = excluded_token_ids.to(chosen_ids.device).long()
        valid &= ~torch.isin(chosen_ids, ex)

    if not valid.any().item():
        return float("nan")
    log_probs_tokens = log_probs[valid, chosen_ids[valid]]
    return np.round(float(torch.exp(log_probs_tokens.mean()).item()), 3)


def _prediction_start_after_tags(new_tokens: torch.Tensor, tag_tensors: Sequence[torch.Tensor]) -> Optional[int]:
    """Index of first token *after* a recognized PREDICTION marker, or ``None``."""
    for tag in tag_tensors:
        n = tag.numel()
        if n == 0 or new_tokens.numel() < n:
            continue
        for k in range(new_tokens.numel() - n, -1, -1):
            if torch.all(new_tokens[k : k + n] == tag):
                return k + n
    return None

def _experiment_timing(
    experiment_number: int, total_experiments: int, start_time: float
) -> tuple[str, str]:
    """Return elapsed and estimated remaining time as H:MM:SS strings."""
    elapsed = time.time() - start_time
    completed = experiment_number - 1
    remaining_secs = (
        (elapsed / completed) * (total_experiments - experiment_number + 1)
        if completed > 0
        else 0.0
    )

    def fmt(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return fmt(elapsed), fmt(remaining_secs)

def save_confusion_matrix(y_true, y_pred, path_txt: str, class_labels: list[str] = ["non-hateful", "hateful"]):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.index.name = "label"
    with open(path_txt, "w") as f:
        f.write("Confusion matrix (rows=true, cols=predicted)\n\n")
        f.write(cm_df.to_string())
        f.write("\n")


def predict(
    dataset: datasets.Dataset,
    model,
    tokenizer,
    system_prompt: str,
    model_kind: Literal["causal", "seq2seq"] = "causal",
    batch_size: int = 8,
    num_batches: int = 1e9,
    generation_config: Optional[dict] = None,
    num_generation_retries: int = 4,
    retry_number: int = 0
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
    sample_ids = []
    problematic_generations = []
    iterator = list(dataset)
    if num_batches == 1e9:
        num_samples = len(iterator)
    else:
        num_samples = num_batches * batch_size

    special_ids = torch.tensor(
        sorted({int(t) for t in getattr(tokenizer, "all_special_ids", []) if t is not None}),
        dtype=torch.long,
    )
    prediction_tag_tensors = [
        torch.tensor(tokenizer.encode(prefix, add_special_tokens=False), dtype=torch.long)
        for prefix in ("PREDICTION:", "\n\nPREDICTION:", "\nPREDICTION:")
    ]

    num_predict_batches = (num_samples + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, num_samples, batch_size),
        desc=f"Predicting (Attempt {retry_number} of {num_generation_retries})",
        total=num_predict_batches,
        miniters=10,
        mininterval=0,
    ):
        batch = iterator[start : start + batch_size]
        batch_labels = [item["label"] for item in batch]
        batch_texts = [item["text"] for item in batch]

        prompt_strings = [
            build_prompt(tokenizer, system_prompt, item["text"])
            for item in batch
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

        with torch.no_grad():
            outputs = model.generate(
                output_scores=True,
                return_dict_in_generate=True,
                **inputs,
                **generation_kwargs,
            )

        prompt_len = 0 if model_kind == "seq2seq" else int(inputs["input_ids"].shape[1])
        sequences = outputs.sequences.detach().cpu()
        scores = torch.stack([s.detach().cpu() for s in outputs.scores], dim=1)
        del outputs, inputs
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        for j in range(len(batch)):
            i = start + j
            row = batch[j]
            row_id = row["id"] if isinstance(row, dict) and "id" in row else i
            new_tokens = sequences[j][prompt_len:]
            pred_start = _prediction_start_after_tags(new_tokens, prediction_tag_tensors)
            if pred_start is None:
                # Fallback to checking if removing the first token helps finding 'PREDICTION'
                prediction_tag_tensors = [p[1:] for p in prediction_tag_tensors]
                pred_start = _prediction_start_after_tags(new_tokens, prediction_tag_tensors)
                if pred_start is None:
                    pred_start = 0

            prediction = tokenizer.decode(new_tokens[pred_start:], skip_special_tokens=True).strip().lower()
            if "non-hateful" in prediction or "non hateful" in prediction or prediction in ("non-hateful", "non hateful"):
                answer = "non-hateful"
            elif prediction == "hateful" or prediction.startswith("hateful") or prediction in ("hate speech", "hate-speech"):
                answer = "hateful"
            else:
                answer = None
            if answer is None:
                problematic_generations.append(
                    {
                        "id": row_id,
                        "text": batch_texts[j],
                        "answer": tokenizer.decode(new_tokens, skip_special_tokens=False),
                        "label": batch_labels[j],
                    }
                )
                continue

            token_offset = 0 if model_kind == "seq2seq" else prompt_len
            confidence_score = calculate_confidence_score(
                scores[j][pred_start:, :],
                sequences[j, token_offset + pred_start :],
                excluded_token_ids=special_ids if special_ids.numel() else None,
            )
            predictions.append(answer)
            confidence_scores.append(confidence_score)
            texts.append(batch_texts[j])
            labels.append(batch_labels[j])
            sample_ids.append(row_id)

    predictions_df = pd.DataFrame(
        {
            "id": sample_ids,
            "text": texts,
            "prediction": predictions,
            "confidence_score": confidence_scores,
            "label": labels,
            "retry_number": retry_number,
        }
    )
    problematic_generations_df = pd.DataFrame(problematic_generations)
    if len(problematic_generations) > 0 and retry_number < num_generation_retries:
        # Enabling sampling so that the model can generate more diverse responses which might help in generating the correct format of the response
        generation_config["do_sample"] = True
        if generation_config.get("temperature") is None or generation_config["temperature"] == 0.0:
            generation_config["temperature"] = 0.8
        if generation_config.get("top_p") is None or generation_config["top_p"] == 1.0:
            generation_config["top_p"] = 0.9
        predictions_df2, problematic_generations_df2 = predict(
            datasets.Dataset.from_pandas(problematic_generations_df),
            model,
            tokenizer,
            system_prompt,
            model_kind=model_kind,
            batch_size=batch_size,
            num_batches=1e9,
            generation_config=generation_config,
            num_generation_retries=num_generation_retries,
            retry_number=retry_number + 1,
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

    datasets_list = []
    for dataset_name in tqdm(config["datasets"], desc="Loading datasets"):
        definitions = []
        for hate_speech_definition in config["datasets"][dataset_name]["hate_speech_definitions"] + config.get("extra_hate_speech_definitions", []):
            try:
                definition = HateSpeechDefinition.load_definition(hate_speech_definition)
            except Exception as e:
                print(f"Error loading definition {hate_speech_definition.get('type')}: {e}")
                continue
            definitions.append(definition)
        try:
            ds_cfg = config["datasets"][dataset_name]
            datasets_list.append(
                HateSpeechDataset.load_dataset(
                    dataset_name,
                    ds_cfg["path"],
                    ds_cfg["text_column"],
                    ds_cfg["label_column"],
                    definitions,
                    id_column=ds_cfg.get("id_column"),
                )
            )
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue
    print("Datasets loaded")
    print(*[f"{dataset.name}: {dataset.dataset.num_rows}" for dataset in datasets_list])

    model_paths = {}
    for model_name in tqdm(config["models"], desc="Locating models paths"):
        try:
            model_paths[model_name] = config["models"][model_name]["path"]
        except Exception as e:
            print(f"Error locating model path for {model_name}: {e}")
            continue
    
    print("Models paths:")
    print(*[f"{name}: {model_paths[name]}" for name in model_paths])

    prompting = []
    for prompting_config in config["prompting"]:
        if prompting_config["type"] == "zero-shot":
            prompting.append(ZeroShotPrompting(name=prompting_config["name"], reasoning_enabled=prompting_config["reasoning_enabled"]))
        elif prompting_config["type"] == "few-shot":
            if "embedding_model_path" in prompting_config:
                try:
                    embedding_model = load_embedding_model(prompting_config["embedding_model_path"], DEVICE)
                except Exception as e:
                    print(f"Error loading embedding model {prompting_config['embedding_model_path']}: {e}")
            else:
                embedding_model = None
            prompting.append(FewShotPrompting(name=prompting_config["name"], reasoning_enabled=prompting_config["reasoning_enabled"], num_shots=prompting_config["num_shots"], few_shot_mode=prompting_config["few_shot_mode"], embedding_model=embedding_model))
        else:
            raise ValueError(f"Invalid prompting type: {prompting_config['type']}")

    print("Prompting methods loaded")
    print(*[f"{prompting_method.name}" for prompting_method in prompting])

    print("-" * 100)
    dataset_names = []
    model_names = []
    prompting_names = []
    definition_names = []
    macro_f1_scores = []
    percentage_of_problematic_generations = []
    total_num_experiments = len(model_paths) * len(prompting) * sum(
        len(dataset.hate_speech_definitions) for dataset in datasets_list
    )
    print(f"Total number of experiments: {total_num_experiments}\n")
    experiment_number = 1
    start_time = time.time()
    for model_path in model_paths:
        current_model, current_tokenizer, model_kind = load_model(model_paths[model_path], DEVICE)
        print(f"Model architecture: {model_kind}")
        for dataset in datasets_list:
            model_name = model_path.replace("/", "_")
            for prompting_method in prompting:
                for definition in dataset.hate_speech_definitions:
                    batch_size = 8
                    while True:
                        experiment_folder = os.path.join(run_folder, dataset.name, model_name, definition.name, prompting_method.name)
                        os.makedirs(experiment_folder)
                        print("-" * 100)
                        elapsed_fmt, remaining_fmt = _experiment_timing(
                            experiment_number, total_num_experiments, start_time
                        )
                        print(
                            f"Running experiment {experiment_number}/{total_num_experiments}:\n"
                            f"Time passed: {elapsed_fmt}, Time remaining (est.): {remaining_fmt}\n"
                            f"model: {model_name}\n"
                            f"dataset: {dataset.name}\n"
                            f"definition: {definition.name}\n"
                            f"prompting: {prompting_method.name}\n"
                        )

                        if isinstance(prompting_method, FewShotPrompting): 
                            system_prompt = prompting_method.build_system_prompt(definition, [(item["text"], item["label"]) for item in list(dataset)])
                        else:
                            system_prompt = prompting_method.build_system_prompt(definition)

                        with open(os.path.join(experiment_folder, "prompt.txt"), "w") as f:
                            f.write(system_prompt)
                        try:
                            predictions_df, problematic_generations_df = predict(
                                dataset,
                                current_model,
                                current_tokenizer,
                                system_prompt,
                                model_kind=model_kind,
                                num_batches=1 if debug_mode else 1e9,
                                batch_size=batch_size,
                                generation_config=generation_config,
                                num_generation_retries=config.get("num_generation_retries", 5),
                            )
                            print("A total of " + str(len(problematic_generations_df)) + " problematic generations were found")
                            print(problematic_generations_df)
                            problematic_generations_df.to_csv(os.path.join(experiment_folder, "problematic_generations.csv"), index=False)
                            text_report = classification_report(
                                predictions_df["label"],
                                predictions_df["prediction"],
                                labels=["non-hateful", "hateful"],
                                target_names=["non-hateful", "hateful"],
                                zero_division=0,
                            )
                            print(text_report)
                            dict_report = classification_report(
                                predictions_df["label"],
                                predictions_df["prediction"],
                                labels=["non-hateful", "hateful"],
                                target_names=["non-hateful", "hateful"],
                                zero_division=0,
                                output_dict=True,
                            )
                            macro_f1_score = np.round(dict_report["macro avg"]["f1-score"], 3)
                            dataset_names.append(dataset.name)
                            model_names.append(model_name)
                            definition_names.append(definition.name)
                            prompting_names.append(prompting_method.name)
                            macro_f1_scores.append(macro_f1_score)
                            percentage_of_problematic_generations.append(np.round(len(problematic_generations_df) / len(predictions_df), 3))
                            with open(os.path.join(experiment_folder, "classification_report.txt"), "w") as f:
                                f.write(text_report)
                            save_confusion_matrix(
                                predictions_df["label"],
                                predictions_df["prediction"],
                                os.path.join(experiment_folder, "confusion_matrix.txt"),
                                class_labels=["non-hateful", "hateful"],
                            )
                            if config["save_predictions"]:
                                predictions_df.to_csv(os.path.join(experiment_folder, "predictions.csv"), index=False)
                            experiment_number += 1
                        except Exception as e:
                            print('X'*100)
                            print(f"Error running experiment {experiment_number}/{total_num_experiments}: {e}")
                            traceback.print_exc()
                       
                            if 'memory' in str(e).lower():
                                print(f"Lowering the batch size to {batch_size // 2}")
                                batch_size = batch_size // 2
                                print('X'*100)
                                continue
                        break

        del current_model, current_tokenizer
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()

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



