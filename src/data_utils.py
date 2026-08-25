"""Load local or Hub datasets and wrap them with associated hate-speech definitions."""

import datasets
from typing import List, Optional

from definitions import HateSpeechDefinition


def _load_raw_dataset(dataset_path: str):
    """Load a dataset from the Hugging Face Hub or a local file, without processing columns."""
    if dataset_path.endswith(".csv"):
        return datasets.load_dataset("csv", data_files=dataset_path)
    elif dataset_path.endswith(".json"):
        return datasets.load_dataset("json", data_files=dataset_path)
    else:
        return datasets.load_dataset(dataset_path)


def _train_split(dataset):
    """Return the train split if present, otherwise the dataset itself."""
    if hasattr(dataset, "keys") and "train" in dataset:
        return dataset["train"]
    return dataset


def _rename_columns(
    dataset: datasets.Dataset,
    text_column: str,
    label_column: str,
    id_column: Optional[str] = None,
) -> datasets.Dataset:
    """Rename text, label, and id columns to standardized names."""
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if label_column != "label":
        dataset = dataset.rename_column(label_column, "label")
    if id_column and id_column != "id":
        dataset = dataset.rename_column(id_column, "id")
    return dataset


class HateSpeechDataset:
    """
    A wrapper class for datasets.DatasetDict or datasets.Dataset with associated hate speech definitions.
    Does NOT subclass datasets.Dataset, so iteration and split access are explicit.
    """

    def __init__(
        self, name: str, dataset, hate_speech_definitions: List[HateSpeechDefinition]
    ):
        """
        Args:
            dataset (datasets.DatasetDict or datasets.Dataset): The loaded and processed dataset object.
            hate_speech_definitions (List[HateSpeechDefinition]): Definitions for hate speech.
        """
        self.name = name
        self.dataset = (
            dataset  # Usually a DatasetDict (for splits), but could be a Dataset
        )
        self.hate_speech_definitions = hate_speech_definitions

    def __getitem__(self, key):
        # Forward access to the underlying dataset object (either by split name or index).
        return self.dataset[key]

    def __iter__(self):
        # Iterate over the 'train' split if it exists, otherwise iterate over the entire Dataset
        if hasattr(self.dataset, "keys") and "train" in self.dataset:
            return iter(self.dataset["train"])
        elif hasattr(self.dataset, "__iter__"):
            return iter(self.dataset)
        else:
            raise AttributeError("Underlying dataset is not iterable.")

    @staticmethod
    def load_dataset(
        name: str,
        dataset_path: str,
        text_column: str,
        label_column: str,
        hate_speech_definitions: List[HateSpeechDefinition],
        id_column: Optional[str] = None,
    ) -> "HateSpeechDataset":
        """Safely load a dataset and wrap it in a HateSpeechDataset with the provided definition."""
        raw_dataset = _load_raw_dataset(dataset_path)
        split = _train_split(raw_dataset)
        processed_dataset = _rename_columns(split, text_column, label_column, id_column)
        return HateSpeechDataset(name, processed_dataset, hate_speech_definitions)
