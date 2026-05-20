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

    TARGET_GROUPS_DOMAIN = ["gender", "sexual orientation", "race", "color", "ethnicity", "nationality", "religion", "disability", "age", "language", "class", "familial status", "pregnancy"]
    DOMINANCE_GROUPS_DOMAIN = ["white_people", "men"]
    PERPETRATOR_CHARACTERISTICS_DOMAIN = ["dominance of group", "societal role", "member of target group"]
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
        if not all(group in CriteriaHateSpeechDefinition.TARGET_GROUPS_DOMAIN for group in self._target_groups):
            raise ValueError(f"Invalid target groups: {self._target_groups}. Expected one of: {CriteriaHateSpeechDefinition.TARGET_GROUPS_DOMAIN}")
        if not all(group in CriteriaHateSpeechDefinition.DOMINANCE_GROUPS_DOMAIN for group in self._dominance_groups):
            raise ValueError(f"Invalid dominance groups: {self._dominance_groups}. Expected one of: {CriteriaHateSpeechDefinition.DOMINANCE_GROUPS_DOMAIN}")
        if not all(characteristic in CriteriaHateSpeechDefinition.PERPETRATOR_CHARACTERISTICS_DOMAIN for characteristic in self._perpetrator_characteristics):
            raise ValueError(f"Invalid perpetrator characteristics: {self._perpetrator_characteristics}. Expected one of: {CriteriaHateSpeechDefinition.PERPETRATOR_CHARACTERISTICS_DOMAIN}")
        if not all(reference in CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN for reference in self._explicit_reference):
            raise ValueError(f"Invalid explicit reference: {self._explicit_reference}. Expected one of: {CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN}")
        if not all(consequence in CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN for consequence in self._effects_consequences):
            raise ValueError(f"Invalid effects/consequences: {self._effects_consequences}. Expected one of: {CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN}")

    @staticmethod
    def _load_definition(
        definition_config: dict,
    ) -> "CriteriaHateSpeechDefinition":
        try:
            name = definition_config["name"]
            target_groups = definition_config["target_groups"]
            dominance = definition_config["dominance"]
            dominance_groups = definition_config["dominance_groups"]
            perpetrator_characteristics = definition_config["perpetrator_characteristics"]
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

        return CriteriaHateSpeechDefinition(name, target_groups, dominance, dominance_groups, perpetrator_characteristics, explicit_reference, insults_group, effects_consequences)

    def prompt_text(self) -> str:
        """
        Render a criteria definition with explicit **inclusions** vs **not required** wording.

        Phrases like ``does not incite X`` read as ``hate speech never involves X``, which is wrong:
        here ``X`` is **out of scope for this benchmark slice**—the label does not *require* X.
        """
        lines: list[str] = []
        excluded_target_groups = sorted(
            set(CriteriaHateSpeechDefinition.TARGET_GROUPS_DOMAIN) - set(self._target_groups)
        )
        if self._target_groups:
            lines.append(
                "Hate speech (under this definition) is abusive or hostile language directed at a person or group "
                f"because of one or more of these traits: {', '.join(self._target_groups)}."
            )
            lines.append(
                "Attacks that target someone only on grounds outside that list (with no clear link to those traits) "
                f"are not hate speech under this definition. Traits outside this list for targeting purposes: "
                f"{', '.join(excluded_target_groups)}."
            )
        else:
            lines.append(
                "Hate speech (under this definition) is not specified via target-group traits in this configuration."
            )

        if self._dominance:
            lines.append(
                "Dominance: both dominant and non-dominant groups can be targets under this definition."
            )
        else:
            lines.append(
                "Dominance: only non-dominant groups are in scope. Abuse aimed solely at dominant groups "
                "(e.g. generic attacks on white people or men as dominant-class stand-ins) is not hate speech "
                "under this definition."
            )
        if self._dominance_groups:
            lines.append(
                f"Dominant groups explicitly named in this definition: {', '.join(self._dominance_groups)}."
            )

        not_required: list[str] = []

        excluded_effects = sorted(
            set(CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN) - set(self._effects_consequences)
        )
        if self._effects_consequences:
            not_required.append(
                f"Incitement: the definition may treat these as relevant when present: "
                f"{', '.join(self._effects_consequences)}. It does not require incitement of: "
                f"{', '.join(excluded_effects)}."
            )
        else:
            not_required.append(
                "Incitement: you do not need discrimination, hate, or violence to be present or incited "
                "for the utterance to count as hateful; hostile targeting on the included traits can be enough."
            )

        excluded_explicit = sorted(
            set(CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN) - set(self._explicit_reference)
        )
        if self._explicit_reference:
            not_required.append(
                f"Explicit reference: the definition may require {', '.join(self._explicit_reference)}. "
                f"It does not require: {', '.join(excluded_explicit)}."
            )
        else:
            not_required.append(
                "Slurs / stereotypes: you do not need a slur, stereotype, or explicit group-characteristic "
                "reference for the hateful label; general hostility toward the group on the included traits can suffice."
            )

        excluded_perp = sorted(
            set(CriteriaHateSpeechDefinition.PERPETRATOR_CHARACTERISTICS_DOMAIN)
            - set(self._perpetrator_characteristics)
        )
        if self._perpetrator_characteristics:
            not_required.append(
                f"Perpetrator: the definition may depend on {', '.join(self._perpetrator_characteristics)}. "
                f"It does not require: {', '.join(excluded_perp)}."
            )
        else:
            not_required.append(
                "Perpetrator: you do not need to infer the speaker's dominance, societal role, or membership "
                "in the target group; the label is about the content toward the target, not those speaker traits."
            )

        if self._insults_group:
            lines.append(
                "Group insult: the definition requires that the language insults the targeted group "
                "(not merely disagrees with an individual in a neutral way)."
            )
        else:
            not_required.append(
                "Group insult: you do not need a direct insult formula (e.g. X are scum); other hostile group "
                "targeting on the included traits can still count."
            )

        lines.append("Not required for the hateful label (do not treat absence as proof of non-hateful):")
        lines.extend(f"- {s}" for s in not_required)

        return "\n".join(lines)

    @staticmethod
    def TYPE() -> str:
        return "criteria" 