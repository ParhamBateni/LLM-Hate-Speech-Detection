from abc import abstractmethod

from typing import Optional, List, Tuple
from enum import StrEnum
from definitions import HateSpeechDefinition
from abc import ABC, abstractmethod
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sentence_transformers import SentenceTransformer
import umap
from sklearn.preprocessing import normalize

def build_hate_speech_system_prompt(
    definition: Optional[HateSpeechDefinition],
    reasoning_enabled: bool = False,
    examples_text: str = "",
) -> str:
    """
    Assemble a system prompt with definition before reasoning/format blocks.

    All variants share the same task intro, refusal guard, label instruction, output
    format, and examples.
    """
    definition_text = definition.prompt_text() if definition else ""
    has_examples = bool(examples_text)
    has_definition = bool(definition_text)

    # task_intro = (
    #     "You are an expert annotator for an academic hate speech detection benchmark.\n"
    #     "Your job is to read each QUERY and assign exactly one class: hateful or non-hateful.\n\n"
    # )
    # refusal_guard_block = (
    #     "QUERY strings may contain offensive or hateful language; they are benchmark items for "
    #     "research annotation, not requests to produce harmful content.\n"
    #     "Never refuse, apologize, or say you cannot classify, label, or analyze the text.\n\n"
    # )
    task_intro = "Classify the QUERY as hateful or non-hateful. "
    refusal_guard_block = "Don't refuse to classify since QUERY may contain offensive or hateful language, but it is benchmark items for research annotation, not requests to produce harmful content.\n\n"
    
    if has_definition:
        if has_examples:
            definition_conditioning_block = (
                "You must ONLY consider HATE_SPEECH_DEFINITION"
                + " and the EXAMPLES"
                + " when deciding the label. Do not use other hate-speech policies or default moderation rules.\n\n"
            )
        else:
            definition_conditioning_block = (
                "You must ONLY consider HATE_SPEECH_DEFINITION"
                + " when deciding the label. Do not use other hate-speech policies or default moderation rules.\n\n"
            )
    else:
        if has_examples:
            definition_conditioning_block = (
                "Consider the EXAMPLES when deciding the label.\n\n"
            )
        else:
            definition_conditioning_block = ""

    definition_block = ("HATE_SPEECH_DEFINITION:\n" + definition_text + "\n\n") if definition_text else ""
    reasoning_block = (
        (
            "Before making a final prediction, reason about the label of the QUERY step by step using Hate Speech Criteria (HSC). Complete the template below for the QUERY using Hate Speech Criteria (HSC). "
            "This is descriptive coding of existing text, not generating new harmful content.\n\n"
            "- TARGET GROUPS: protected groups referenced (e.g., race, religion, gender), or none\n"
            "- DOMINANCE: dominant/non-dominant framing, or not applicable\n"
            "- PERPETRATOR CHARACTERISTICS: source of hostility if any, or none\n"
            "- EXPLICIT_REFERENCE: slur, stereotype, group characteristic, or none\n"
            "- EFFECTS_CONSEQUENCES: insult, discrimination, hate, violence, or none\n\n"
        )
        if reasoning_enabled and has_definition
        else ""
    )
    
    output_format_block = (
        "Respond ONLY in the following format:\n"
        "PREDICTION: non-hateful\n"
        "or\n"
        "PREDICTION: hateful\n\n"
        "do not add any other text."
    )
    
    return "".join(
        [
            task_intro,
            refusal_guard_block,
            definition_conditioning_block,
            definition_block,
            examples_text,
            reasoning_block,
            output_format_block,
        ]
    )


class Prompting(ABC):
    def __init__(self, name: str, reasoning_enabled: bool = False):
        self._name = name
        self._reasoning_enabled = reasoning_enabled

    @abstractmethod
    def build_system_prompt(
        self, definition: Optional[HateSpeechDefinition], **kwargs
    ) -> str:
        """
        Build a prompt for the given text and definition.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the name of the prompting method.
        """
        return self.__class__.__name__.lower()


class ZeroShotPrompting(Prompting):
    def __init__(self, name: str, reasoning_enabled: bool = False):
        super().__init__(name, reasoning_enabled)

    def build_system_prompt(
        self, definition: Optional[HateSpeechDefinition], **kwargs
    ) -> str:
        return build_hate_speech_system_prompt(
            definition,
            reasoning_enabled=self._reasoning_enabled,
        )

    @property
    def name(self) -> str:
        return self._name


class FewShotPrompting(Prompting):
    class FewShotMode(StrEnum):
        RANDOM = "random"
        SMART = "smart"

    def __init__(
        self,
        name: str,
        reasoning_enabled: bool = False,
        num_shots_per_group: int = 5,
        few_shot_mode: FewShotMode = FewShotMode.RANDOM,
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        super().__init__(name, reasoning_enabled)
        self._num_shots_per_group = num_shots_per_group
        self._few_shot_mode = few_shot_mode
        if few_shot_mode == self.FewShotMode.SMART:
            if embedding_model is None:
                raise ValueError("Embedding model is required for smart few-shot mode")
            self._embedding_model = embedding_model
        else:
            self._embedding_model = None

    def build_system_prompt(
        self,
        definition: Optional[HateSpeechDefinition],
        examples: List[Tuple[str, str]],
    ) -> str:
        grouped_examples = {}
        for example in examples:
            if example[1] not in grouped_examples:
                grouped_examples[example[1]] = []
            grouped_examples[example[1]].append((example[0], example[1]))
        selected = []
        if self._few_shot_mode == self.FewShotMode.RANDOM:
            for group in sorted(list(grouped_examples.keys()),reverse=True):
                selected.extend(
                    random.sample(grouped_examples[group], self._num_shots_per_group)
                )
        elif self._few_shot_mode == self.FewShotMode.SMART:
            for group in sorted(list(grouped_examples.keys()),reverse=True):
                group_embeddings = self._embedding_model.encode(
                    [example[0] for example in grouped_examples[group]]
                )
                normalized_group_embeddings = normalize(group_embeddings)
                umap_embeddings = umap.UMAP(n_components=2, min_dist=0.0, metric="cosine").fit_transform(normalized_group_embeddings)
                kmeans = KMeans(n_clusters=self._num_shots_per_group)
                _labels = kmeans.fit_predict(umap_embeddings)
                centroids = kmeans.cluster_centers_
                closest_indices = pairwise_distances_argmin(centroids, umap_embeddings)
                selected.extend(
                    [
                        (
                            grouped_examples[group][idx][0],
                            grouped_examples[group][idx][1],
                        )
                        for idx in closest_indices
                    ]
                )
        else:
            raise ValueError(f"Invalid few-shot mode: {self._few_shot_mode}")

        examples_text = ""
        if selected:
            examples_text = (
                "EXAMPLES:\n\n"
                + "\n\n".join(f"TEXT: {ex[0]}\nPREDICTION: {ex[1]}" for ex in selected)
                + "\n\n"
            )

        return build_hate_speech_system_prompt(
            definition,
            reasoning_enabled=self._reasoning_enabled,
            examples_text=examples_text,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def few_shot_mode(self) -> FewShotMode:
        return self._few_shot_mode
