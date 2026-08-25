"""Vanilla (free-text) and criteria-based (HSC) hate-speech definitions for prompts."""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
import json


class HateSpeechDefinition(ABC):
    """
    Abstract base class for hate speech definitions.

    Concrete subclasses implement ``prompt_text`` (the human-readable definition
    inserted into model prompts) and a static ``_load_definition`` factory used by
    :meth:`load_definition`.
    """

    def __init__(self, name: str):
        self._name = name

    @staticmethod
    def load_definition(definition_spec: dict[str, Any]) -> "HateSpeechDefinition":
        """
        Factory method. ``definition_spec`` may be either:
          - a dict containing a ``path`` key pointing to a JSON file with the full config
          - a dict containing the full config inline (e.g. for vanilla definitions)
        """
        if "path" in definition_spec and definition_spec["path"]:
            with open(definition_spec["path"], "r") as f:
                definition_config = json.load(f)
            for key in ("type", "name"):
                if key in definition_spec:
                    definition_config[key] = definition_spec[key]
        else:
            definition_config = definition_spec

        definition_type = definition_config.get("type") or definition_spec.get("type")
        if definition_type == VanillaHateSpeechDefinition.type_name():
            return VanillaHateSpeechDefinition._load_definition(definition_config)
        domain_config = None
        domain_path = definition_spec.get("domain_path")
        if domain_path:
            with open(domain_path, "r") as f:
                domain_config = json.load(f)
        exclude_aspects = definition_spec.get("exclude_aspects")
        if definition_type == CriteriaHateSpeechDefinition.type_name():
            return CriteriaHateSpeechDefinition._load_definition(
                definition_config,
                domain_config=domain_config,
                exclude_aspects=exclude_aspects,
            )
        raise ValueError(
            f"Invalid hate speech definition type: {definition_type!r}. Expected one of: "
            f"{VanillaHateSpeechDefinition.type_name()!r}, "
            f"{CriteriaHateSpeechDefinition.type_name()!r}, "
        )

    @staticmethod
    @abstractmethod
    def _load_definition(definition_config: dict) -> "HateSpeechDefinition":
        """Construct a concrete definition from a config dict."""

    @abstractmethod
    def prompt_text(self) -> str:
        """Return the hate speech definition as a human-readable prompt string."""

    @staticmethod
    @abstractmethod
    def type_name() -> str:
        """Return the string discriminator of the definition type (``vanilla`` or ``criteria``)."""

    @property
    def name(self) -> str:
        return self._name


class VanillaHateSpeechDefinition(HateSpeechDefinition):
    """Custom hate speech definition."""

    def __init__(self, name: str, definition_text: str):
        super().__init__(name)
        self._definition_text = definition_text

    @staticmethod
    def _load_definition(definition_config: dict) -> "VanillaHateSpeechDefinition":
        try:
            name = definition_config["name"]
            definition_text = definition_config["definition_text"]
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(
                f"Missing required key '{missing_key}' in VanillaHateSpeechDefinition configuration. "
                "Expected keys: ['name', 'definition_text'].\n"
                f"Offending config: {definition_config}"
            ) from e
        return VanillaHateSpeechDefinition(name, definition_text)

    def prompt_text(self) -> str:
        return self._definition_text

    @staticmethod
    def type_name() -> str:
        return "vanilla"


class CriteriaHateSpeechDefinition(HateSpeechDefinition):
    """
    A hate speech definition specified through the HSC aspect template.

    The set of aspects is driven entirely by the :class:`Domain` instance passed in:
    adding or removing aspects from the domain file flows through automatically to
    validation and to :meth:`prompt_text` rendering, without code changes here.
    """

    def __init__(
        self,
        name: str,
        target_groups: List[str],
        dominance: bool,
        dominant_groups: List[str],
        perpetrator_characteristics: List[str],
        explicit_reference: List[str],
        insults_group: bool,
        incites: List[str],
        domain_config: Optional[dict] = None,
        exclude_aspects: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self._target_groups = target_groups
        self._dominance = dominance
        self._dominant_groups = dominant_groups
        self._perpetrator_characteristics = perpetrator_characteristics
        self._explicit_reference = explicit_reference
        self._insults_group = insults_group
        self._incites = incites
        self._domain_config = domain_config
        self._exclude_aspects = exclude_aspects
        if domain_config:
            self._validate()

    def _validate(self):
        if "others" not in self._domain_config["target_groups"]["domain"] and not all(
            group in self._domain_config["target_groups"]["domain"]
            for group in self._target_groups
        ):
            raise ValueError(
                f"Invalid target groups: {self._target_groups}. Expected one of: {self._domain_config['target_groups']['domain']}"
            )
        if "others" not in self._domain_config["dominant_groups"]["domain"] and not all(
            group in self._domain_config["dominant_groups"]["domain"]
            for group in self._dominant_groups
        ):
            raise ValueError(
                f"Invalid dominance groups: {self._dominant_groups}. Expected one of: {self._domain_config['dominant_groups']['domain']}"
            )
        if "others" not in self._domain_config["perpetrator_characteristics"][
            "domain"
        ] and not all(
            characteristic
            in self._domain_config["perpetrator_characteristics"]["domain"]
            for characteristic in self._perpetrator_characteristics
        ):
            raise ValueError(
                f"Invalid perpetrator characteristics: {self._perpetrator_characteristics}. Expected one of: {self._domain_config['perpetrator_characteristics']['domain']}"
            )
        if "others" not in self._domain_config["explicit_reference"][
            "domain"
        ] and not all(
            reference in self._domain_config["explicit_reference"]["domain"]
            for reference in self._explicit_reference
        ):
            raise ValueError(
                f"Invalid explicit reference: {self._explicit_reference}. Expected one of: {self._domain_config['explicit_reference']['domain']}"
            )
        if "others" not in self._domain_config["incites"]["domain"] and not all(
            incite in self._domain_config["incites"]["domain"]
            for incite in self._incites
        ):
            raise ValueError(
                f"Invalid incites: {self._incites}. Expected one of: {self._domain_config['incites']['domain']}"
            )

    @staticmethod
    def _load_definition(
        definition_config: dict,
        domain_config: Optional[dict] = None,
        exclude_aspects: Optional[List[str]] = None,
    ) -> "CriteriaHateSpeechDefinition":
        try:
            name = definition_config["name"]
            target_groups = definition_config["target_groups"]
            dominance = definition_config["dominance"]
            dominant_groups = definition_config["dominant_groups"]
            perpetrator_characteristics = definition_config[
                "perpetrator_characteristics"
            ]
            explicit_reference = definition_config["explicit_reference"]
            insults_group = definition_config["insults_group"]
            incites = definition_config["incites"]
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(
                f"Missing required key '{missing_key}' in CriteriaHateSpeechDefinition configuration. "
                "Expected keys: ['name', 'target_groups', 'dominance', 'dominance_groups', 'perpetrator_characteristics', 'explicit_reference', 'insults_group', 'effects_consequences'].\n"
                f"Offending config: {definition_config}"
            ) from e

        return CriteriaHateSpeechDefinition(
            name,
            target_groups,
            dominance,
            dominant_groups,
            perpetrator_characteristics,
            explicit_reference,
            insults_group,
            incites,
            domain_config,
            exclude_aspects,
        )

    @staticmethod
    def _format_list(items: List[str], conjunction: str = "or") -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"

    def _domain_excluded_phrases(
        self,
        aspect_key: str,
        selected: List[str],
        labels: Optional[dict[str, str]] = None,
    ) -> List[str]:
        """Domain values omitted from ``selected`` (excluding the ``others`` sentinel)."""
        if not self._domain_config or aspect_key not in self._domain_config:
            return []
        domain = self._domain_config[aspect_key]["domain"]
        omitted = [
            item for item in domain if item not in selected and item not in {"others"}
        ]
        labels = labels or {}
        return [labels.get(item, str(item)) for item in omitted]

    def _apply_domain_exclusions(
        self,
        text: str,
        aspect_key: str,
        selected: List[str],
        labels: Optional[dict[str, str]] = None,
    ) -> str:
        if not self._exclude_aspects or aspect_key not in self._exclude_aspects:
            return text

        excluded_phrases = self._domain_excluded_phrases(aspect_key, selected, labels)
        if not excluded_phrases:
            return text

        labels = {
            "target_groups": "Target groups",
            "dominant_groups": "Dominant groups",
            "perpetrator_characteristics": "Perpetrator characteristics",
            "explicit_reference": "Explicit references",
            "incites": "Effects",
        }
        standalone = (
            f"{labels.get(aspect_key, aspect_key.replace('_', ' '))} such as "
            f"{self._format_list(excluded_phrases)} are not considered"
        )
        if text:
            return text.rstrip(". ") + f". {standalone}"
        return standalone + "."

    def _target_groups_phrase(self) -> str:
        if not self._target_groups:
            return ""
        text = f" based on their {self._format_list(self._target_groups)}"
        return self._apply_domain_exclusions(text, "target_groups", self._target_groups)

    def _dominance_phrase(self) -> str:
        if self._dominance:
            if self._dominant_groups:
                groups = self._format_list(self._dominant_groups)
                text = f"person or group (including {groups})"
                return self._apply_domain_exclusions(
                    text, "dominant_groups", self._dominant_groups
                )
            return "person or group"
        return "historically non-dominant person or group"

    def _perpetrator_characteristics_phrase(self) -> str:
        characteristic_labels = {
            "dominance of group": "dominance of group",
            "member of target group": "member of target group",
            "societal role": "societal role",
        }
        phrases = []
        for characteristic in self._perpetrator_characteristics:
            if characteristic == "member of target group":
                phrases.append(
                    "Consider whether the speaker is a member of the targeted group, as self-directed references or reclaimed language may not constitute hate speech depending on the intent and context."
                )
            elif characteristic == "dominance of group":
                phrases.append(
                    "Consider the relative social power between the speaker and the targeted group, as hateful statements directed from a socially dominant group toward a marginalized group are generally more harmful than the reverse."
                )
            elif characteristic == "societal role":
                phrases.append(
                    "Consider the speaker's societal role, as statements made by individuals with authority or public influence (e.g., politicians, executives, or public figures) may have greater impact than identical statements made by private individuals."
                )
        text = " ".join(phrases) if phrases else ""
        return self._apply_domain_exclusions(
            text,
            "perpetrator_characteristics",
            self._perpetrator_characteristics,
            characteristic_labels,
        )

    def _explicit_reference_phrase(self) -> str:
        if not self._explicit_reference:
            return ""
        phrases = {
            "group characteristic": "group characteristics (including the name of the group)",
            "slur": "slurs",
            "stereotype": "stereotypes",
        }
        reference_phrases = [
            phrases.get(reference, reference) for reference in self._explicit_reference
        ]
        text = f" using references such as {self._format_list(reference_phrases)}"
        return self._apply_domain_exclusions(
            text, "explicit_reference", self._explicit_reference, phrases
        )

    def _incites_phrase(self) -> str:
        incite_labels = {
            "discrimination": "discrimination",
            "hate": "hate",
            "violence": "violence",
        }
        if self._incites:
            incite_part = f"incites {self._format_list(self._incites)}"
            incite_part = self._apply_domain_exclusions(
                incite_part, "incites", self._incites, incite_labels
            )
            if self._insults_group:
                return f"{incite_part}, or insults a group"
            return incite_part
        if self._insults_group:
            return "insults a group"
        return "expresses hostility toward a group"

    def prompt_text(self) -> str:
        """Render the criteria definition as a compact natural-language paragraph."""
        base = "Hate speech is defined as language targeted at a "
        dominance = self._dominance_phrase()
        target_groups = self._target_groups_phrase()
        incites = self._incites_phrase()
        explicit_reference = self._explicit_reference_phrase()
        perpetrator = self._perpetrator_characteristics_phrase()
        text = (
            f"{base}{dominance}{target_groups}. It {incites}"
            f"{' on the basis of aforementioned targets' if target_groups else ''}"
            f"{explicit_reference}. "
            f"{perpetrator}"
        )
        return text

    @staticmethod
    def type_name() -> str:
        return "criteria"
