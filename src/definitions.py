from abc import ABC, abstractmethod
from typing import List, Any
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
        if definition_type == VanillaHateSpeechDefinition.TYPE():
            return VanillaHateSpeechDefinition._load_definition(definition_config)
        if definition_type == CriteriaHateSpeechDefinition.TYPE():
            return CriteriaHateSpeechDefinition._load_definition(definition_config)
        raise ValueError(
            f"Invalid hate speech definition type: {definition_type!r}. Expected one of: "
            f"{VanillaHateSpeechDefinition.TYPE()!r}, "
            f"{CriteriaHateSpeechDefinition.TYPE()!r}, "
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
    def TYPE() -> str:
        """Return the string discriminator of the definition type."""

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
    def TYPE() -> str:
        return "vanilla"


class CriteriaHateSpeechDefinition(HateSpeechDefinition):
    """
    A hate speech definition specified through the HSC aspect template.

    The set of aspects is driven entirely by the :class:`Domain` instance passed in:
    adding or removing aspects from the domain file flows through automatically to
    validation and to :meth:`prompt_text` rendering, without code changes here.
    """

    TARGET_GROUPS_DOMAIN = [
        "gender",
        "sexual orientation",
        "race",
        "color",
        "ethnicity",
        "nationality",
        "religion",
        "disability",
        "age",
        "language",
        "class",
        "familial status",
        "pregnancy",
    ]
    DOMINANCE_GROUPS_DOMAIN = ["white_people", "men"]
    PERPETRATOR_CHARACTERISTICS_DOMAIN = [
        "dominance of group",
        "societal role",
        "member of target group",
    ]
    EXPLICIT_REFERENCE_DOMAIN = ["stereotype", "group characteristic", "slur"]
    EFFECTS_CONSEQUENCES_DOMAIN = ["violence", "hate", "discrimination"]

    def __init__(
        self,
        name: str,
        target_groups: List[str],
        dominance: bool,
        dominance_groups: List[str],
        perpetrator_characteristics: List[str],
        explicit_reference: List[str],
        insults_group: bool,
        effects_consequences: List[str],
    ):
        super().__init__(name)
        self._target_groups = target_groups
        self._dominance = dominance
        self._dominance_groups = dominance_groups
        self._perpetrator_characteristics = perpetrator_characteristics
        self._explicit_reference = explicit_reference
        self._insults_group = insults_group
        self._effects_consequences = effects_consequences
        self._validate()

    def _validate(self):
        if not all(
            group in CriteriaHateSpeechDefinition.TARGET_GROUPS_DOMAIN
            for group in self._target_groups
        ):
            raise ValueError(
                f"Invalid target groups: {self._target_groups}. Expected one of: {CriteriaHateSpeechDefinition.TARGET_GROUPS_DOMAIN}"
            )
        if not all(
            group in CriteriaHateSpeechDefinition.DOMINANCE_GROUPS_DOMAIN
            for group in self._dominance_groups
        ):
            raise ValueError(
                f"Invalid dominance groups: {self._dominance_groups}. Expected one of: {CriteriaHateSpeechDefinition.DOMINANCE_GROUPS_DOMAIN}"
            )
        if not all(
            characteristic
            in CriteriaHateSpeechDefinition.PERPETRATOR_CHARACTERISTICS_DOMAIN
            for characteristic in self._perpetrator_characteristics
        ):
            raise ValueError(
                f"Invalid perpetrator characteristics: {self._perpetrator_characteristics}. Expected one of: {CriteriaHateSpeechDefinition.PERPETRATOR_CHARACTERISTICS_DOMAIN}"
            )
        if not all(
            reference in CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN
            for reference in self._explicit_reference
        ):
            raise ValueError(
                f"Invalid explicit reference: {self._explicit_reference}. Expected one of: {CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN}"
            )
        if not all(
            consequence in CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN
            for consequence in self._effects_consequences
        ):
            raise ValueError(
                f"Invalid effects/consequences: {self._effects_consequences}. Expected one of: {CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN}"
            )

    @staticmethod
    def _load_definition(
        definition_config: dict,
    ) -> "CriteriaHateSpeechDefinition":
        try:
            name = definition_config["name"]
            target_groups = definition_config["target_groups"]
            dominance = definition_config["dominance"]
            dominance_groups = definition_config["dominance_groups"]
            perpetrator_characteristics = definition_config[
                "perpetrator_characteristics"
            ]
            explicit_reference = definition_config["explicit_reference"]
            insults_group = definition_config["insults_group"]
            effects_consequences = definition_config["effects_consequences"]
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
            dominance_groups,
            perpetrator_characteristics,
            explicit_reference,
            insults_group,
            effects_consequences,
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

    def _dominance_phrase(self) -> str:
        if self._dominance:
            if self._dominance_groups:
                groups = self._format_list(self._dominance_groups)
                return f"person or group (including {groups})"
            return "person or group"
        return "historically non-dominant person or group"

    def _ordered_subset(
        self, selected: List[str], presentation_order: List[str]
    ) -> List[str]:
        ordered = [item for item in presentation_order if item in selected]
        remaining = [item for item in selected if item not in presentation_order]
        return ordered + remaining

    def _effects_consequences_phrase(self) -> str:
        clauses: list[str] = []
        effects = self._ordered_subset(
            self._effects_consequences,
            CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN,
        )
        if effects:
            clauses.append(f"incites {self._format_list(effects)}")
        if self._insults_group:
            clauses.append("insults a group")
        if clauses:
            return " or ".join(clauses)
        return "expresses hostility toward a group"

    def _explicit_reference_phrase(self) -> str:
        references = self._ordered_subset(
            self._explicit_reference,
            CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN,
        )
        if not references:
            return ""
        phrases = {
            "group characteristic": "group characteristics (including the name of the group)",
            "slur": "slurs",
            "stereotype": "stereotypes",
        }
        reference_phrases = [
            phrases.get(reference, reference) for reference in references
        ]
        return f" using references such as {self._format_list(reference_phrases)}"

    def _target_groups_phrase(self) -> str:
        if self._target_groups:
            return f" based on their {self._format_list(self._target_groups)}"
        return ""

    def prompt_text(self) -> str:
        """Render the criteria definition as a compact natural-language paragraph."""
        base = "Hate speech is defined as language targeted at a "
        dominance = self._dominance_phrase()
        target_groups = self._target_groups_phrase()
        effects_consequences = self._effects_consequences_phrase()
        explicit_reference = self._explicit_reference_phrase()
        return f"{base}{dominance}{target_groups}. It {effects_consequences}{' on the basis of aforementioned targets' if target_groups else ''}{explicit_reference}."

    @staticmethod
    def TYPE() -> str:
        return "criteria"
