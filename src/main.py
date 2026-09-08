"""Run the config-driven experiment grid: load models, prompt, generate, and score."""

import gc
import os
import shutil
import time
import traceback

import datasets
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from typing import Literal, Optional

from generation_utils import (
    build_prediction_tag_tensors,
    build_prompt,
    calculate_confidence_score,
    get_special_token_ids,
    parse_prediction_label,
    prediction_start_after_tags,
)
from config_utils import (
    load_datasets_from_config,
    load_model_paths_from_config,
    load_prompting_from_config,
    read_config,
)
from data_utils import HateSpeechDataset
from definitions import HateSpeechDefinition
from model_utils import clear_cache_directory, load_model, resolve_device, seed_everything
from prompting import FewShotPrompting, Prompting

load_dotenv()


def predict(
    dataset: datasets.Dataset,
    model,
    tokenizer,
    prompting_method: Prompting,
    definition: HateSpeechDefinition,
    device: torch.device,
    model_kind: Literal["causal", "seq2seq"] = "causal",
    batch_size: int = 8,
    num_batches: int = 1e9,
    generation_config: Optional[dict] = None,
    compute_confidence_score: bool = False,
    num_generation_retries: int = 4,
    retry_number: int = 0,
    few_shot_examples: Optional[list[tuple[str, str]]] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generation_config = dict(generation_config or {})
    max_new_tokens = generation_config.get("max_new_tokens", 100)
    do_sample = generation_config.get("do_sample", False)
    temperature = generation_config.get("temperature", 1.0)
    top_p = generation_config.get("top_p", 1.0)

    texts, predictions, confidence_scores, labels, sample_ids = [], [], [], [], []
    problematic_generations = []
    iterator = list(dataset)
    num_samples = len(iterator) if num_batches == 1e9 else num_batches * batch_size

    if isinstance(prompting_method, FewShotPrompting) and few_shot_examples is None:
        few_shot_examples = [(item["text"], item["label"]) for item in iterator]

    special_ids = get_special_token_ids(tokenizer)
    prediction_tag_tensors = build_prediction_tag_tensors(tokenizer)
    num_predict_batches = (num_samples + batch_size - 1) // batch_size

    # Nearest-query few-shot depends on the current text, so the system prompt is rebuilt per item.
    if (
        isinstance(prompting_method, FewShotPrompting)
        and prompting_method.few_shot_mode == FewShotPrompting.FewShotMode.NEAREST_QUERY
    ):
        system_prompt = None
    elif isinstance(prompting_method, FewShotPrompting):
        system_prompt = prompting_method.build_system_prompt(
            definition=definition,
            examples=few_shot_examples,
            random_state=seed,
            use_cache=True,
        )
    else:
        system_prompt = prompting_method.build_system_prompt(definition)

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

        if system_prompt is None:
            prompt_strings = [
                build_prompt(
                    tokenizer,
                    prompting_method.build_system_prompt(
                        definition=definition,
                        examples=few_shot_examples,
                        random_state=seed,
                        query=item["text"],
                        use_cache=True,
                    ),
                    item["text"],
                )
                for item in batch
            ]
        else:
            prompt_strings = [
                build_prompt(tokenizer, system_prompt, item["text"]) for item in batch
            ]

        inputs = tokenizer(
            prompt_strings, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"], device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
                output_scores=compute_confidence_score,
                return_dict_in_generate=True,
                **generation_kwargs,
            )

        prompt_len = 0 if model_kind == "seq2seq" else int(inputs["input_ids"].shape[1])
        sequences = outputs.sequences.detach().cpu()
        if compute_confidence_score:
            scores = torch.stack([s.detach().cpu() for s in outputs.scores], dim=1)
        del outputs, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for j, row in enumerate(batch):
            i = start + j
            row_id = row["id"] if isinstance(row, dict) and "id" in row else i
            new_tokens = sequences[j][prompt_len:]
            pred_start = prediction_start_after_tags(new_tokens, prediction_tag_tensors)
            if pred_start is None:
                fallback_tags = [p[1:] for p in prediction_tag_tensors]
                pred_start = prediction_start_after_tags(new_tokens, fallback_tags)
                if pred_start is None:
                    pred_start = 0

            prediction = (
                tokenizer.decode(new_tokens[pred_start:], skip_special_tokens=True)
                .strip()
                .lower()
            )
            answer = parse_prediction_label(prediction)
            if answer is None:
                problematic_generations.append(
                    {
                        "id": row_id,
                        "text": batch_texts[j],
                        "answer": tokenizer.decode(
                            new_tokens, skip_special_tokens=False
                        ),
                        "label": batch_labels[j],
                    }
                )
                continue

            if compute_confidence_score:
                token_offset = 0 if model_kind == "seq2seq" else prompt_len
                score = calculate_confidence_score(
                    scores[j][pred_start:, :],
                    sequences[j, token_offset + pred_start :],
                    excluded_token_ids=special_ids if special_ids.numel() else None,
                )
            else:
                score = None
            predictions.append(answer)
            confidence_scores.append(score)
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

    # Unparseable outputs are retried with sampling so a later decode can hit the expected format.
    if len(problematic_generations) > 0 and retry_number < num_generation_retries:
        generation_config["do_sample"] = True
        if (
            generation_config.get("temperature") is None
            or generation_config["temperature"] == 0.0
        ):
            generation_config["temperature"] = 0.8
        if generation_config.get("top_p") is None or generation_config["top_p"] == 1.0:
            generation_config["top_p"] = 0.9
        predictions_df2, problematic_generations_df2 = predict(
            datasets.Dataset.from_pandas(problematic_generations_df),
            model,
            tokenizer,
            prompting_method,
            definition=definition,
            device=device,
            model_kind=model_kind,
            batch_size=batch_size,
            num_batches=1e9,
            generation_config=generation_config,
            compute_confidence_score=compute_confidence_score,
            num_generation_retries=num_generation_retries,
            retry_number=retry_number + 1,
            few_shot_examples=few_shot_examples,
            seed=seed,
        )
        predictions_df = pd.concat([predictions_df, predictions_df2])
        problematic_generations_df = problematic_generations_df2

    return predictions_df, problematic_generations_df


def _experiment_timing(
    experiment_number: int, total_experiments: int, start_time: float
) -> tuple[str, str]:
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


def _save_system_prompt(
    prompting_method: Prompting,
    definition: HateSpeechDefinition,
    dataset: HateSpeechDataset,
    experiment_folder: str,
    seed: int,
) -> None:
    should_save = (
        isinstance(prompting_method, FewShotPrompting)
        and prompting_method.few_shot_mode != FewShotPrompting.FewShotMode.NEAREST_QUERY
    ) or not isinstance(prompting_method, FewShotPrompting)
    if not should_save:
        return

    if isinstance(prompting_method, FewShotPrompting):
        system_prompt = prompting_method.build_system_prompt(
            definition=definition,
            examples=[(item["text"], item["label"]) for item in list(dataset)],
            random_state=seed,
            use_cache=True,
        )
    else:
        system_prompt = prompting_method.build_system_prompt(definition)

    with open(os.path.join(experiment_folder, "prompt.txt"), "w") as f:
        f.write(system_prompt)


def _save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    path_txt: str,
    class_labels: list[str] | None = None,
) -> None:
    if class_labels is None:
        class_labels = ["non-hateful", "hateful"]
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.index.name = "label"
    with open(path_txt, "w") as f:
        f.write("Confusion matrix (rows=true, cols=predicted)\n\n")
        f.write(cm_df.to_string())
        f.write("\n")


if __name__ == "__main__":
    run_id = time.strftime("%m_%d_%H:%M:%S")
    run_folder = os.path.join("runs", run_id)
    os.makedirs(run_folder)
    print("Run ID: " + run_id)

    print("CUDA available: " + str(torch.cuda.is_available()))
    print("MPS available: " + str(torch.backends.mps.is_available()))
    device = resolve_device()
    print("Using device: " + str(device))

    config = read_config("config.yaml")
    shutil.copy("config.yaml", os.path.join(run_folder, "config.yaml"))
    print("Debug mode: " + str(config["debug_mode"]))

    generation_config = config.get("generation", {})
    seed = generation_config.get("seed", 42)
    seed_everything(seed)
    print(f"Seed set to: {seed}")
    print(f"Generation config: {generation_config}")

    datasets_list = load_datasets_from_config(config)
    print("Datasets loaded")
    print(*[f"{d.name}: {d.dataset.num_rows}" for d in datasets_list])

    model_paths = load_model_paths_from_config(config)
    print("Models paths:")
    print(*[f"{name}: {model_paths[name]}" for name in model_paths])

    prompting = load_prompting_from_config(config, device)
    print("Prompting methods loaded")
    print(*[f"{p.name}" for p in prompting])

    dataset_names, model_names, prompting_names, definition_names = [], [], [], []
    macro_f1_scores, percentage_of_problematic_generations = [], []
    total_num_experiments = (
        len(model_paths)
        * len(prompting)
        * sum(len(d.hate_speech_definitions) for d in datasets_list)
    )
    print("-" * 100)
    print(f"Total number of experiments: {total_num_experiments}\n")

    experiment_number = 1
    start_time = time.time()
    for model_path in model_paths:
        # Clear cache directory to free up space to prevent out of disk space error.
        clear_cache_directory()
        current_model, current_tokenizer, model_kind = load_model(
            model_paths[model_path], device
        )
        print(f"Model architecture: {model_kind}")
        model_name = model_path.replace("/", "_")

        for dataset in datasets_list:
            for prompting_method in prompting:
                for definition in dataset.hate_speech_definitions:
                    batch_size = 8
                    # Retry the same experiment with a smaller batch if generation OOMs.
                    while True:
                        experiment_folder = os.path.join(
                            run_folder,
                            dataset.name,
                            model_name,
                            definition.name,
                            prompting_method.name,
                        )
                        os.makedirs(experiment_folder, exist_ok=True)
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
                        _save_system_prompt(
                            prompting_method,
                            definition,
                            dataset,
                            experiment_folder,
                            seed,
                        )
                        try:
                            predictions_df, problematic_generations_df = predict(
                                dataset,
                                current_model,
                                current_tokenizer,
                                prompting_method,
                                definition=definition,
                                device=device,
                                model_kind=model_kind,
                                num_batches=1 if config["debug_mode"] else 1e9,
                                batch_size=batch_size,
                                generation_config=generation_config,
                                compute_confidence_score=config.get(
                                    "compute_confidence_score", False
                                ),
                                num_generation_retries=config.get(
                                    "num_generation_retries", 4
                                ),
                                seed=seed,
                            )
                            print(
                                f"A total of {len(problematic_generations_df)} "
                                "problematic generations were found"
                            )
                            print(problematic_generations_df)
                            problematic_generations_df.to_csv(
                                os.path.join(
                                    experiment_folder, "problematic_generations.csv"
                                ),
                                index=False,
                            )
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
                            dataset_names.append(dataset.name)
                            model_names.append(model_name)
                            definition_names.append(definition.name)
                            prompting_names.append(prompting_method.name)
                            macro_f1_scores.append(
                                np.round(dict_report["macro avg"]["f1-score"], 3)
                            )
                            percentage_of_problematic_generations.append(
                                np.round(
                                    len(problematic_generations_df)
                                    / len(predictions_df),
                                    3,
                                )
                            )
                            with open(
                                os.path.join(
                                    experiment_folder, "classification_report.txt"
                                ),
                                "w",
                            ) as f:
                                f.write(text_report)
                            _save_confusion_matrix(
                                predictions_df["label"],
                                predictions_df["prediction"],
                                os.path.join(experiment_folder, "confusion_matrix.txt"),
                            )
                            if config["save_predictions"]:
                                predictions_df.to_csv(
                                    os.path.join(experiment_folder, "predictions.csv"),
                                    index=False,
                                )
                            experiment_number += 1
                        except Exception as e:
                            print("X" * 100)
                            print(
                                f"Error running experiment "
                                f"{experiment_number}/{total_num_experiments}: {e}"
                            )
                            traceback.print_exc()
                            if "memory" in str(e).lower():
                                print(f"Lowering the batch size to {batch_size // 2}")
                                batch_size = batch_size // 2
                                print("X" * 100)
                                continue
                        break

        del current_model, current_tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
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
