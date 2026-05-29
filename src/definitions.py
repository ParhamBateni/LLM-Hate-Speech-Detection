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
    def _aspect_line(aspect: str, side: str, text: str) -> str:
        return f"{aspect}. {side}: {text}"

    def prompt_text(self) -> str:
        """
        Render a criteria definition with explicit **inclusions** vs **not required** wording.

        Phrases like ``does not incite X`` read as ``hate speech never involves X``, which is wrong:
        here ``X`` is **out of scope for this benchmark slice**—the label does not *require* X.
        """
        lines: list[str] = []
        aspect = self._aspect_line
        lines.append(
            "Hate speech (under this definition) is abusive or hostile language; "
            "a sample is hateful only if it satisfies the included criteria below."
        )

        excluded_target_groups = sorted(
            set(CriteriaHateSpeechDefinition.TARGET_GROUPS_DOMAIN)
            - set(self._target_groups)
        )
        if self._target_groups:
            lines.append(
                aspect(
                    "Target groups",
                    "Included (hateful samples must target someone on at least one of these traits)",
                    ", ".join(self._target_groups),
                )
            )
            if excluded_target_groups:
                lines.append(
                    aspect(
                        "Target groups",
                        "Excluded (targeting only on these grounds, with no clear link to included traits, is not hateful)",
                        ", ".join(excluded_target_groups),
                    )
                )
        else:
            lines.append(
                aspect(
                    "Target groups",
                    "Included",
                    "not specified in this configuration",
                )
            )
            if excluded_target_groups:
                lines.append(
                    aspect(
                        "Target groups",
                        "Excluded",
                        (
                            ", ".join(excluded_target_groups)
                            if excluded_target_groups
                            else "none listed"
                        ),
                    )
                )

        if self._dominance:
            included_dominance = "both dominant and non-dominant groups"
            if self._dominance_groups:
                included_dominance += f"; dominant groups explicitly named: {', '.join(self._dominance_groups)}"
            lines.append(
                aspect(
                    "Dominance",
                    "Included (hateful samples may target)",
                    included_dominance,
                )
            )
        else:
            lines.append(
                aspect(
                    "Dominance",
                    "Included (hateful samples may target)",
                    "non-dominant groups only",
                )
            )
            excluded_dominance = (
                "abuse aimed solely at dominant groups "
                "(e.g. generic attacks on white people or men as dominant-class stand-ins)"
            )
            if self._dominance_groups:
                excluded_dominance += f"; named dominant groups out of scope: {', '.join(self._dominance_groups)}"
            if excluded_dominance:
                lines.append(
                    aspect(
                        "Dominance",
                        "Excluded (not hateful under this definition)",
                        excluded_dominance,
                    )
                )

        excluded_effects = sorted(
            set(CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN)
            - set(self._effects_consequences)
        )
        if self._effects_consequences:
            lines.append(
                aspect(
                    "Effects/consequences",
                    "Included (hateful samples must include at least one of)",
                    ", ".join(self._effects_consequences),
                )
            )
            if excluded_effects:
                lines.append(
                    aspect(
                        "Effects/consequences",
                        "Excluded (not required for the hateful label)",
                        ", ".join(excluded_effects) if excluded_effects else "none",
                    )
                )
        else:
            lines.append(
                aspect(
                    "Effects/consequences",
                    "Included (sufficient for hateful)",
                    "hostile targeting on the included traits",
                )
            )
            lines.append(
                aspect(
                    "Effects/consequences",
                    "Excluded (not required)",
                    ", ".join(CriteriaHateSpeechDefinition.EFFECTS_CONSEQUENCES_DOMAIN),
                )
            )

        excluded_explicit = sorted(
            set(CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN)
            - set(self._explicit_reference)
        )
        if self._explicit_reference:
            lines.append(
                aspect(
                    "Explicit reference",
                    "Included (hateful samples must include at least one of)",
                    ", ".join(self._explicit_reference),
                )
            )
            if excluded_explicit:
                lines.append(
                    aspect(
                        "Explicit reference",
                        "Excluded (not required for the hateful label)",
                        ", ".join(excluded_explicit) if excluded_explicit else "none",
                    )
                )
        else:
            lines.append(
                aspect(
                    "Explicit reference",
                    "Included (sufficient for hateful)",
                    "general hostility toward the group on the included traits",
                )
            )
            lines.append(
                aspect(
                    "Explicit reference",
                    "Excluded (not required)",
                    ", ".join(CriteriaHateSpeechDefinition.EXPLICIT_REFERENCE_DOMAIN),
                )
            )

        excluded_perp = sorted(
            set(CriteriaHateSpeechDefinition.PERPETRATOR_CHARACTERISTICS_DOMAIN)
            - set(self._perpetrator_characteristics)
        )
        if self._perpetrator_characteristics:
            lines.append(
                aspect(
                    "Perpetrator",
                    "Included (hateful label depends on the speaker's)",
                    ", ".join(self._perpetrator_characteristics),
                )
            )
            if excluded_perp:
                lines.append(
                    aspect(
                        "Perpetrator",
                        "Excluded (not required for the hateful label)",
                        ", ".join(excluded_perp) if excluded_perp else "none",
                    )
                )
        else:
            lines.append(
                aspect(
                    "Perpetrator",
                    "Included (what matters for the label)",
                    "the content toward the target, not speaker traits",
                )
            )
            lines.append(
                aspect(
                    "Perpetrator",
                    "Excluded (not required)",
                    ", ".join(
                        CriteriaHateSpeechDefinition.PERPETRATOR_CHARACTERISTICS_DOMAIN
                    ),
                )
            )

        if self._insults_group:
            lines.append(
                aspect(
                    "Group insult",
                    "Included (required for hateful)",
                    "language that insults the targeted group (not merely disagrees with an individual in a neutral way)",
                )
            )
            lines.append(
                aspect(
                    "Group insult",
                    "Excluded (not sufficient for hateful)",
                    "neutral disagreement with an individual without group-directed insult",
                )
            )
        else:
            lines.append(
                aspect(
                    "Group insult",
                    "Included (sufficient for hateful)",
                    "other hostile group targeting on the included traits",
                )
            )
            lines.append(
                aspect(
                    "Group insult",
                    "Excluded (not required)",
                    "a direct insult formula (e.g. X are scum)",
                )
            )

        return "\n".join(lines)

    @staticmethod
    def TYPE() -> str:
        return "criteria"
