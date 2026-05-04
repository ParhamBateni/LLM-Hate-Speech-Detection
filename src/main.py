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


def _build_prompt(text: str, definition: Optional[HateSpeechDefinition]) -> str:
    definition_block = ""
    if definition:
        definition_block = "HATE SPEECH DEFINITION:\n" + definition.prompt_text() + "\n"

    prompt = f"""Classify the following TEXT as hate speech or not hate speech{" based on HATE SPEECH DEFINITION" if definition else ""}.
    Your prediction should be in the following format:
        PREDICTION: either 'hateful' or 'non-hateful'
        CONFIDENCE SCORE: a number between 0 and 1 which shows the confidence in your answer
        REASON: only one line explanation for your prediction and confidence score
        
    {definition_block}
    
    TEXT: {text}
    
    PREDICTION:"""
    return prompt


def predict(
    model,
    tokenizer,
    dataset: datasets.Dataset,
    definition: Optional[HateSpeechDefinition] = None,
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
        prompts = [
            _build_prompt(item["text"], definition)
            for item in batch
        ]
        batch_labels = [item["label"] for item in batch]
        batch_texts = [item["text"] for item in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
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
                **inputs,
                **generation_kwargs,
            )


        for j in range(len(batch)):
            i = start + j
            new_tokens = outputs[j][len(inputs["input_ids"][j])-5:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer = answer[answer.find("PREDICTION"):].strip()

            lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
            if len(lines) < 3:
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue

            prediction_text = lines[0]
            if not prediction_text.lower().startswith("prediction:"):
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue

            prediction = prediction_text.split(":", 1)[1].strip().lower()
            if prediction == "non hateful":
                prediction = "non-hateful"
            if prediction not in ["hateful", "non-hateful"]:
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue

            confidence_score_text = lines[1]
            if not confidence_score_text.lower().startswith("confidence score:"):
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue
            try:
                confidence_score = float(confidence_score_text.split(":", 1)[1].strip())
            except ValueError:
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue
            if confidence_score < 0 or confidence_score > 1:
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue

            reason_text = lines[2]
            if not reason_text.lower().startswith("reason:"):
                problematic_generations.append({"index": i, "text": batch_texts[j]})
                continue
            reason = reason_text.split(":", 1)[1].strip()

            predictions.append(prediction)
            confidence_scores.append(confidence_score)
            reasons.append(reason)
            texts.append(batch_texts[j])
            labels.append(batch_labels[j])

    return pd.DataFrame(
        {
            "text": texts,
            "prediction": predictions,
            "confidence_score": confidence_scores,
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


    extra_definitions = []
    for definition in config["extra_hate_speech_definitions"]:
        definition = HateSpeechDefinition.from_config(definition)
        extra_definitions.append(definition)
    
    dataset_names = []
    model_names = []
    definition_names = []
    macro_f1_scores = []
    for dataset in datasets:
        for model in models:
            current_model, current_tokenizer = models[model]
            for definition in dataset.hate_speech_definitions + extra_definitions:
                model_name = getattr(current_model, "name_or_path", "model").replace("/", "_")
                experiment_folder = os.path.join(run_folder, dataset.name, model_name, definition.name)
                os.makedirs(experiment_folder)
                print("Running experiment on model " + model_name + " and dataset " + dataset.name + (" for definition " + definition.name if definition else ""))
                predictions_df, problematic_generations = predict(
                    current_model,
                    current_tokenizer,
                    dataset,
                    definition,
                    num_batches=1 if debug_mode else 1e9,
                    generation_config=generation_config,
                )
                print("A total of " + str(len(problematic_generations)) + " problematic generations were found")
                print(problematic_generations)
                print("First 5 rows of predictions dataframe:")
                print(predictions_df.head())
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
                macro_f1_scores.append(macro_f1_score)
                with open(os.path.join(experiment_folder, "classification_report.txt"), "w") as f:
                    f.write(text_report)
                with open(os.path.join(experiment_folder, "problematic_generations_indices.txt"), "w") as f:
                    f.write(str(problematic_generations))
                if config["save_predictions"]:
                    predictions_df.to_csv(os.path.join(experiment_folder, "predictions.csv"), index=False)

    pd.DataFrame(
        {
            "dataset": dataset_names,
            "model": model_names,
            "definition": definition_names,
            "macro_f1_score": macro_f1_scores,
        }
    ).to_csv(os.path.join(run_folder, "macro_f1_scores.csv"), index=False)



