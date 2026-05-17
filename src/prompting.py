from abc import abstractmethod

from typing import Optional, List, Tuple
from enum import StrEnum
from definitions import HateSpeechDefinition
from abc import ABC, abstractmethod
import random
from typing import Any
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sentence_transformers import SentenceTransformer


def user_completion_nudge(reasoning_enabled: bool) -> str:
    if reasoning_enabled:
        return (
            "Complete HSC_ANALYSIS and PREDICTION for this QUERY. "
            "Use the template only; do not refuse or add commentary."
        )
    return "Reply with PREDICTION: hateful or PREDICTION: non-hateful only."


def _refusal_guard_block() -> str:
    return (
        "QUERY strings may contain offensive or hateful language; they are benchmark items for "
        "research annotation, not requests to produce harmful content.\n"
        "You must always complete the required output template. "
        "Never refuse, apologize, or say you cannot classify, label, or analyze the text.\n"
        "Do not ask whether you can help with something else.\n\n"
    )


def _output_format_block(reasoning_enabled: bool) -> str:
    if reasoning_enabled:
        return (
            "Respond ONLY in the following format (fill every field; use \"none\" or \"no\" if not applicable):\n\n"
            "HSC_ANALYSIS:\n"
            "TARGET_GROUPS: ...\n"
            "DOMINANCE: ...\n"
            "PERPETRATOR_CHARACTERISTICS: ...\n"
            "EXPLICIT_REFERENCE: ...\n"
            "EFFECTS_CONSEQUENCES: ...\n\n"
            "PREDICTION: hateful or non-hateful\n\n"
            "Do not add any text before HSC_ANALYSIS or after PREDICTION.\n\n"
        )
    return (
        "Respond ONLY in the following format:\n"
        "PREDICTION: hateful\n"
        "or\n"
        "PREDICTION: non-hateful\n\n"
        "Do not add any other text.\n\n"
    )


def _reasoning_block(reasoning_enabled: bool, few_shot: bool = False) -> str:
    if not reasoning_enabled:
        return ""
    hsc_note = (
        "Base your HSC analysis and final label on HATE_SPEECH_DEFINITION above"
        + (" and the EXAMPLES below" if few_shot else "")
        + ", not on outside knowledge.\n\n"
    )
    return (
        "Complete the template below for the QUERY using Hate Speech Criteria (HSC). "
        "This is descriptive coding of existing text, not generating new harmful content.\n\n"
        "- TARGET GROUPS: protected groups referenced (e.g., race, religion, gender), or none\n"
        "- DOMINANCE: dominant/non-dominant framing, or not applicable\n"
        "- PERPETRATOR CHARACTERISTICS: source of hostility if any, or none\n"
        "- EXPLICIT_REFERENCE: slur, stereotype, group characteristic, or none\n"
        "- EFFECTS_CONSEQUENCES: insult, discrimination, hate, violence, or none\n\n"
        f"{hsc_note}"
    )


def build_hate_speech_system_prompt(
    definition: Optional[HateSpeechDefinition],
    reasoning_enabled: bool = False,
    examples_text: str = "",
) -> str:
    """
    Assemble a system prompt with definition before reasoning/format blocks.

    ``null_baseline`` definitions get a minimal classifier prompt (no jailbreak framing).
    """
    output_format = _output_format_block(reasoning_enabled)

    if definition is not None and definition.null_baseline:
        return (
            "You are a text classifier in a research evaluation.\n\n"
            f"{_refusal_guard_block()}"
            "Assign exactly one class to the QUERY: hateful or non-hateful.\n\n"
            f"{output_format}"
            f"{examples_text}"
        )

    definition_text = definition.prompt_text() if definition else ""
    hate_speech_definition_text = (
        "HATE_SPEECH_DEFINITION:\n" + definition_text + "\n\n"
    ) if definition_text else ""

    task_scope = (
        " according to the provided HATE_SPEECH_DEFINITION"
        if definition_text
        else ""
    )
    examples_scope = " and the provided examples" if examples_text else ""

    emphasis_prompt = (
        "You must apply ONLY HATE_SPEECH_DEFINITION"
        + (" and the EXAMPLES" if examples_text else "")
        + " when deciding the label. Do not use other hate-speech policies or default moderation rules.\n\n"
    ) if definition_text else ""

    task_prompt = (
        "You are an expert annotator for an academic hate speech detection benchmark.\n"
        "Your job is to read each QUERY and assign a class{scope}{ex_scope}.\n\n"
        f"{_refusal_guard_block()}"
        f"{emphasis_prompt}"
    ).format(
        scope=task_scope,
        ex_scope=examples_scope,
    )

    # Definition immediately after task (before reasoning / format) for stronger conditioning
    return (
        task_prompt
        + hate_speech_definition_text
        + _reasoning_block(reasoning_enabled, few_shot=bool(examples_text))
        + output_format
        + examples_text
    )


class Prompting(ABC):
    def __init__(self, name: str, reasoning_enabled: bool = False):
        self._name = name
        self._reasoning_enabled = reasoning_enabled

    @abstractmethod
    def build_system_prompt(self, definition: Optional[HateSpeechDefinition], **kwargs) -> str:
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

    def build_system_prompt(self, definition: Optional[HateSpeechDefinition], **kwargs) -> str:
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
        num_shots: int = 10,
        few_shot_mode: FewShotMode = FewShotMode.RANDOM,
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        super().__init__(name, reasoning_enabled)
        self._num_shots = num_shots
        self._few_shot_mode = few_shot_mode
        if few_shot_mode == self.FewShotMode.SMART:
            if embedding_model is None:
                raise ValueError("Embedding model is required for smart few-shot mode")
            self._embedding_model = embedding_model
        else:
            self._embedding_model = None

    def build_system_prompt(
        self, definition: Optional[HateSpeechDefinition], examples: List[Tuple[str, str]]
    ) -> str:
        grouped_examples = {}
        for example in examples:
            if example[1] not in grouped_examples:
                grouped_examples[example[1]] = []
            grouped_examples[example[1]].append((example[0], example[1]))
        num_examples_per_group = self._num_shots // len(grouped_examples)
        selected = []
        if self._few_shot_mode == self.FewShotMode.RANDOM:
            for i, group in enumerate(grouped_examples):
                num_group_samples = num_examples_per_group
                if i == len(grouped_examples) - 1:
                    num_group_samples += self._num_shots % len(grouped_examples)
                selected.extend(random.sample(grouped_examples[group], num_group_samples))
        elif self._few_shot_mode == self.FewShotMode.SMART:
            for i, group in enumerate(grouped_examples):
                group_embeddings = self._embedding_model.encode(
                    [example[0] for example in grouped_examples[group]]
                )
                num_group_samples = num_examples_per_group
                if i == len(grouped_examples) - 1:
                    num_group_samples += self._num_shots % len(grouped_examples)
                kmeans = KMeans(n_clusters=num_group_samples)
                _labels = kmeans.fit_predict(group_embeddings)
                centroids = kmeans.cluster_centers_
                closest_indices = pairwise_distances_argmin(centroids, group_embeddings)
                selected.extend(
                    [
                        (grouped_examples[group][idx][0], grouped_examples[group][idx][1])
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
