import datasets
from typing import List
from definitions import HateSpeechDefinition

def _load_raw_dataset(dataset_path: str):
    """Load a dataset from the Hugging Face Hub or a local file, without processing columns."""
    if dataset_path.endswith(".csv"):
        return datasets.load_dataset("csv", data_files=dataset_path)
    elif dataset_path.endswith(".json"):
        return datasets.load_dataset("json", data_files=dataset_path)
    else:
        return datasets.load_dataset(dataset_path)

def _rename_columns(dataset: datasets.Dataset, text_column: str, label_column: str):
    """Rename text and label columns to standardized names."""
    dataset = dataset.rename_column(text_column, "text")
    dataset = dataset.rename_column(label_column, "label")
    return dataset

class HateSpeechDataset:
    """
    A wrapper class for datasets.DatasetDict or datasets.Dataset with associated hate speech definitions.
    Does NOT subclass datasets.Dataset, so iteration and split access are explicit.
    """
    def __init__(self, name: str, dataset, hate_speech_definitions: List[HateSpeechDefinition]):
        """
        Args:
            dataset (datasets.DatasetDict or datasets.Dataset): The loaded and processed dataset object.
            hate_speech_definitions (List[HateSpeechDefinition]): Definitions for hate speech.
        """
        self.name = name
        self.dataset = dataset  # Usually a DatasetDict (for splits), but could be a Dataset
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
    def load_dataset(name: str, dataset_path: str, text_column: str, label_column: str, hate_speech_definitions: List[HateSpeechDefinition]) -> "HateSpeechDataset":
        """Safely load a dataset and wrap it in a HateSpeechDataset with the provided definition."""
        raw_dataset = _load_raw_dataset(dataset_path)
        processed_dataset = _rename_columns(raw_dataset, text_column, label_column)
        return HateSpeechDataset(name, processed_dataset, hate_speech_definitions)