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
        if domain_config:
            self._validate()

    def _validate(self):
        if not all(
            group in self._domain_config["target_groups"]["domain"]
            for group in self._target_groups
        ):
            raise ValueError(
                f"Invalid target groups: {self._target_groups}. Expected one of: {self._domain_config['target_groups']['domain']}"
            )
        if not all(
            group in self._domain_config["dominance_groups"]["domain"]
            for group in self._dominant_groups
        ):
            raise ValueError(
                f"Invalid dominance groups: {self._dominant_groups}. Expected one of: {self._domain_config['dominance_groups']['domain']}"
            )
        if not all(
            characteristic
            in self._domain_config["perpetrator_characteristics"]["domain"]
            for characteristic in self._perpetrator_characteristics
        ):
            raise ValueError(
                f"Invalid perpetrator characteristics: {self._perpetrator_characteristics}. Expected one of: {self._domain_config['perpetrator_characteristics']['domain']}"
            )
        if not all(
            reference in self._domain_config["explicit_reference"]["domain"]
            for reference in self._explicit_reference
        ):
            raise ValueError(
                f"Invalid explicit reference: {self._explicit_reference}. Expected one of: {self._domain_config['explicit_reference']['domain']}"
            )
        if not all(
            incite in self._domain_config["incites"]["domain"]
            for incite in self._incites
        ):
            raise ValueError(
                f"Invalid incites: {self._incites}. Expected one of: {self._domain_config['incites']['domain']}"
            )

    @staticmethod
    def _load_definition(
        definition_config: dict, domain_config: Optional[dict] = None
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

    def _target_groups_phrase(self) -> str:
        if self._target_groups:
            return f" based on their {self._format_list(self._target_groups)}"
        return ""

    def _dominance_phrase(self) -> str:
        if self._dominance:
            if self._dominant_groups:
                groups = self._format_list(self._dominant_groups)
                return f"person or group (including {groups})"
            return "person or group"
        return "historically non-dominant person or group"

    def _perpetrator_characteristics_phrase(self) -> str:
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
        return " ".join(phrases) if phrases else ""

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
        return f" using references such as {self._format_list(reference_phrases)}"

    def _incites_phrase(self) -> str:
        clauses: list[str] = []
        if self._incites:
            clauses.append(f"incites {self._format_list(self._incites)}")
        if self._insults_group:
            clauses.append("insults a group")
        if clauses:
            return " or ".join(clauses)
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
    def TYPE() -> str:
        return "criteria"
