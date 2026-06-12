from abc import abstractmethod

from typing import Optional, List, Tuple
from enum import StrEnum
from definitions import HateSpeechDefinition
from abc import ABC, abstractmethod
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
from itertools import zip_longest


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
    refusal_guard_block = "Don't refuse to classify since QUERY may contain offensive or hateful language, but it is benchmark items for research annotation, not requests to produce harmful content.\n"

    if has_definition:
        if has_examples:
            definition_conditioning_block = (
                "You must ONLY consider HATE_SPEECH_DEFINITION"
                + " and the EXAMPLES"
                + " when deciding the label. Do not use other hate-speech policies or default moderation rules. If you can't confidently classify the QUERY as hateful based on the HATE_SPEECH_DEFINITION and the EXAMPLES, then predict 'non-hateful'.\n\n"
            )
        else:
            definition_conditioning_block = (
                "You must ONLY consider HATE_SPEECH_DEFINITION"
                + " when deciding the label. Do not use other hate-speech policies or default moderation rules. If you can't confidently classify the QUERY as hateful based on the HATE_SPEECH_DEFINITION, then predict 'non-hateful'.\n\n"
            )
    else:
        if has_examples:
            definition_conditioning_block = (
                "Consider the EXAMPLES when deciding the label. If you can't confidently classify the QUERY as hateful based on the EXAMPLES, then predict 'non-hateful'.\n\n"
            )
        else:
            definition_conditioning_block = ""

    definition_block = (
        ("HATE_SPEECH_DEFINITION:\n" + definition_text + "\n\n")
        if definition_text
        else ""
    )
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
        "Respond ONLY in the following format and add no other text:\n"
        "PREDICTION: non-hateful\n"
        "or\n"
        "PREDICTION: hateful\n\n"
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
        DIVERSE = "diverse"
        NEAREST_QUERY = "nearest_query"

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
        if few_shot_mode == self.FewShotMode.DIVERSE:
            if embedding_model is None:
                raise ValueError(
                    "Embedding model is required for diverse few-shot mode"
                )
            self._embedding_model = embedding_model
        elif few_shot_mode == self.FewShotMode.NEAREST_QUERY:
            if embedding_model is None:
                raise ValueError(
                    "Embedding model is required for nearest query few-shot mode"
                )
            self._embedding_model = embedding_model
        else:
            self._embedding_model = None
        self._cache = {}

    def build_system_prompt(
        self,
        definition: Optional[HateSpeechDefinition],
        examples: Optional[List[Tuple[str, str]]] = None,
        random_state: int = 42,
        query: str = None,
        use_cache: bool = False,
    ) -> str:
        if use_cache:
            grouped_examples = self._cache.get("grouped_examples", {})
            if not grouped_examples and not examples:
                raise ValueError("Examples are not given and not cached")
        if examples is not None:
            grouped_examples = {}
            for example in examples:
                if example[1] not in grouped_examples:
                    grouped_examples[example[1]] = []
                grouped_examples[example[1]].append(example[0])
            if (
                use_cache
                and self._cache.get("grouped_examples", {}) != grouped_examples
            ):
                self._cache = {}
            self._cache["grouped_examples"] = grouped_examples

        selected = {}
        if self._few_shot_mode == self.FewShotMode.RANDOM:
            for group in sorted(list(grouped_examples.keys()), reverse=True):
                rng = random.Random(random_state)
                selected[group] = [
                    (example, group)
                    for example in rng.sample(
                        grouped_examples[group], self._num_shots_per_group
                    )
                ]

        elif self._few_shot_mode == self.FewShotMode.DIVERSE:
            if use_cache:
                centroids = self._cache.get("centroids", {})
                normalized_embeddings = self._cache.get("normalized_embeddings", {})
            else:
                centroids = {}
                normalized_embeddings = {}
            for group in sorted(list(grouped_examples.keys()), reverse=True):
                group_examples = grouped_examples[group]
                if group not in centroids or group not in normalized_embeddings:
                    group_embeddings = self._embedding_model.encode(group_examples)
                    normalized_group_embeddings = normalize(group_embeddings)
                    kmeans = KMeans(
                        n_clusters=self._num_shots_per_group,
                        random_state=random_state,
                    )
                    _labels = kmeans.fit_predict(normalized_group_embeddings)
                    group_centroids = kmeans.cluster_centers_
                    centroids[group] = group_centroids
                    normalized_embeddings[group] = normalized_group_embeddings
                else:
                    normalized_group_embeddings = normalized_embeddings[group]
                    group_centroids = centroids[group]
                closest_indices = pairwise_distances_argmin(
                    group_centroids, normalized_group_embeddings
                )
                selected[group] = [
                    (grouped_examples[group][idx], group) for idx in closest_indices
                ]

            self._cache["centroids"] = centroids
            self._cache["normalized_embeddings"] = normalized_embeddings
        elif self._few_shot_mode == self.FewShotMode.NEAREST_QUERY:
            if query is None:
                raise ValueError("Query is required for nearest query few-shot mode")
            query_embedding = self._embedding_model.encode([query])
            normalized_query_embedding = normalize(query_embedding)
            if use_cache:
                normalized_embeddings = self._cache.get("normalized_embeddings", {})
            else:
                normalized_embeddings = {}
            for group in sorted(list(grouped_examples.keys()), reverse=True):
                if group not in normalized_embeddings:
                    group_embeddings = self._embedding_model.encode(
                        grouped_examples[group]
                    )
                    normalized_group_embeddings = normalize(group_embeddings)
                    normalized_embeddings[group] = normalized_group_embeddings
                else:
                    normalized_group_embeddings = normalized_embeddings[group]
                index_query = -1
                try:
                    index_query = grouped_examples[group].index(query)
                except ValueError:
                    pass
                closest_indices = np.argsort(
                    np.linalg.norm(
                        normalized_group_embeddings - normalized_query_embedding, axis=1
                    )
                )[: self._num_shots_per_group + 1]
                closest_indices = [
                    idx for idx in closest_indices if idx != index_query
                ][: self._num_shots_per_group]
                selected[group] = [
                    (grouped_examples[group][idx], group) for idx in closest_indices
                ]

            self._cache["normalized_embeddings"] = normalized_embeddings
        else:
            raise ValueError(f"Invalid few-shot mode: {self._few_shot_mode}")

        if selected:
            interleaved_examples = [
                x
                for pair in zip_longest(*selected.values())
                for x in pair
                if x is not None
            ]
            examples_text = ""
            for i, ex in enumerate(interleaved_examples):
                escaped_text = ex[0].replace('"', "'")
                examples_text += (
                    f"EXAMPLE {i+1}:\n"
                    f'TEXT: "{escaped_text}"\n'
                    f"LABEL: {ex[1]}\n\n"
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
