import torch
import yaml
from tqdm import tqdm

from data_utils import HateSpeechDataset
from definitions import HateSpeechDefinition
from model_utils import load_embedding_model
from prompting import FewShotPrompting, Prompting, ZeroShotPrompting


def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_datasets_from_config(config: dict) -> list[HateSpeechDataset]:
    datasets_list = []
    for dataset_name in tqdm(config["datasets"], desc="Loading datasets"):
        definitions = []
        for hate_speech_definition in config["datasets"][dataset_name].get(
            "hate_speech_definitions", []
        ) + config.get("extra_hate_speech_definitions", []):
            try:
                definitions.append(
                    HateSpeechDefinition.load_definition(hate_speech_definition)
                )
            except Exception as e:
                print(
                    f"Error loading definition {hate_speech_definition.get('type')}: {e}"
                )
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
    return datasets_list


def load_model_paths_from_config(config: dict) -> dict[str, str]:
    model_paths = {}
    for model_name in tqdm(config["models"], desc="Locating models paths"):
        try:
            model_paths[model_name] = config["models"][model_name]["path"]
        except Exception as e:
            print(f"Error locating model path for {model_name}: {e}")
    return model_paths


def load_prompting_from_config(config: dict, device: torch.device) -> list[Prompting]:
    prompting = []
    for prompting_config in config["prompting"]:
        if prompting_config["type"] == "zero-shot":
            prompting.append(
                ZeroShotPrompting(
                    name=prompting_config["name"],
                    reasoning_enabled=prompting_config["reasoning_enabled"],
                )
            )
        elif prompting_config["type"] == "few-shot":
            embedding_model = None
            if "embedding_model_path" in prompting_config:
                try:
                    embedding_model = load_embedding_model(
                        prompting_config["embedding_model_path"], device
                    )
                except Exception as e:
                    print(
                        f"Error loading embedding model "
                        f"{prompting_config['embedding_model_path']}: {e}"
                    )
            prompting.append(
                FewShotPrompting(
                    name=prompting_config["name"],
                    reasoning_enabled=prompting_config["reasoning_enabled"],
                    num_shots_per_group=prompting_config["num_shots_per_group"],
                    few_shot_mode=prompting_config["few_shot_mode"],
                    embedding_model=embedding_model,
                )
            )
        else:
            raise ValueError(f"Invalid prompting type: {prompting_config['type']}")
    return prompting
