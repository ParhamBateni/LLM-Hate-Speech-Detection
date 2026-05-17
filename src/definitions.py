from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Any
import json


class Domain:
    """
    Modular loader and validator for the Hate Speech Criteria (HSC) domain template.

    Structure of a domain file (e.g. data/definitions/domain.json):

      { "aspects": {
            <aspect>: {
                "description": <str, optional>,
                # Leaf form (no sub-aspects):
                "multi_value": <bool>,
                "domain": [<allowed value 1>, <allowed value 2>, ...]
                # OR Branch form (sub-aspects):
                <sub-aspect>: { ... leaf or branch ... },
                ...
            },
            ...
        }
      }

    A leaf node is detected by the presence of both ``multi_value`` and ``domain`` keys.
    Wherever the special token ``"others"`` appears in a leaf's ``domain`` list, the leaf is
    treated as open: any string value is then accepted for that aspect.
    """

    LEAF_KEYS = frozenset({"multi_value", "domain", "open_domain"})
    UNSPECIFIED_LABEL = "unspecified"

    def __init__(
        self,
        aspects: dict,
        description: Optional[str] = None,
    ):
        if not isinstance(aspects, dict) or not aspects:
            raise ValueError("Domain 'aspects' must be a non-empty dict.")
        self._aspects = aspects
        self._description = description

    @staticmethod
    def load(path) -> "Domain":
        path = Path(path)
        with open(path, "r") as f:
            config = json.load(f)
        aspects = config.get("aspects")
        if not isinstance(aspects, dict) or not aspects:
            raise ValueError(f"Domain file '{path}' must contain a non-empty 'aspects' dict.")
        return Domain(
            aspects=aspects,
            description=config.get("$description"),
        )

    @property
    def aspects(self) -> dict:
        return self._aspects

    @property
    def description(self) -> Optional[str]:
        return self._description

    @classmethod
    def is_leaf(cls, node) -> bool:
        return isinstance(node, dict) and cls.LEAF_KEYS.issubset(node.keys())

    @classmethod
    def _iter_branches(cls, node: dict) -> Iterator[Tuple[str, dict]]:
        """Yield (key, child) pairs whose child is a dict (skipping string metadata such as 'description')."""
        for key, child in node.items():
            if isinstance(child, dict):
                yield key, child

    def iter_leaves(
        self,
        node: Optional[dict] = None,
        path: Tuple[str, ...] = (),
    ) -> Iterator[Tuple[Tuple[str, ...], dict]]:
        """Recursively yield ``(path, leaf_node)`` pairs in domain order."""
        if node is None:
            node = self._aspects
        for key, child in self._iter_branches(node):
            current_path = path + (key,)
            if self.is_leaf(child):
                yield current_path, child
            else:
                yield from self.iter_leaves(child, current_path)

    @staticmethod
    def extract_selected(def_value):
        """
        Extract the actual selected value from a definition node.

        Definition files may store leaf values in either of two shapes:
          - direct scalar / list / None (e.g. ``"enable": null``)
          - wrapped dict with a ``selected`` key (e.g. ``{"selected": [...], "notes": "..."}``)
        """
        if isinstance(def_value, dict) and "selected" in def_value:
            return def_value["selected"]
        return def_value

    def validate(self, definition_aspects: dict) -> List[str]:
        """
        Check that ``definition_aspects`` mirrors this domain and that every leaf
        value falls inside the corresponding allowed-values list.

        Returns a list of human-readable error messages; empty list means the
        definition is consistent with the domain.
        """
        errors: List[str] = []
        self._validate_node(self._aspects, definition_aspects, (), errors)
        return errors

    def _validate_node(
        self,
        domain_node: dict,
        def_node,
        path: Tuple[str, ...],
        errors: List[str],
    ) -> None:
        path_str = ".".join(path) if path else "<root>"
        if not isinstance(def_node, dict):
            errors.append(
                f"Aspect '{path_str}' must be a dict to mirror the domain branch, got {type(def_node).__name__}."
            )
            return
        for key, domain_child in self._iter_branches(domain_node):
            current_path = path + (key,)
            if key not in def_node:
                errors.append(f"Missing aspect '{'.'.join(current_path)}' in the definition.")
                continue
            def_child = def_node[key]
            if self.is_leaf(domain_child):
                self._validate_leaf(domain_child, def_child, current_path, errors)
            else:
                self._validate_node(domain_child, def_child, current_path, errors)

    def _validate_leaf(
        self,
        domain_leaf: dict,
        def_value,
        path: Tuple[str, ...],
        errors: List[str],
    ) -> None:
        path_str = ".".join(path)
        multi_value = bool(domain_leaf.get("multi_value", False))
        open_domain = bool(domain_leaf.get("open_domain", False))
        allowed = list(domain_leaf.get("domain", []))
        selected = self.extract_selected(def_value)
        if selected is None:
            return

        if multi_value:
            if not isinstance(selected, list):
                errors.append(
                    f"Aspect '{path_str}' is multi-valued, expected a list (or null), got "
                    f"{type(selected).__name__}: {selected!r}."
                )
                return
            offending = []
            for v in selected:
                if not isinstance(v, str):
                    errors.append(f"Aspect '{path_str}' contains a non-string value: {v!r}.")
                    continue
                if (not open_domain) and v not in allowed:
                    offending.append(v)
            if offending:
                errors.append(
                    f"Aspect '{path_str}' contains values {offending} not in the domain {allowed}. "
                )
            return

        if isinstance(selected, (list, dict)):
            errors.append(
                f"Aspect '{path_str}' is single-valued, expected a scalar (or null), got "
                f"{type(selected).__name__}: {selected!r}."
            )
            return
        if (not open_domain) and selected not in allowed:
            errors.append(
                f"Aspect '{path_str}' value {selected!r} is not in the domain {allowed}. "
            )


class HateSpeechDefinition(ABC):
    """
    Abstract base class for hate speech definitions.

    Concrete subclasses implement ``prompt_text`` (the human-readable definition
    inserted into model prompts) and a static ``_load_definition`` factory used by
    :meth:`load_definition`.
    """

    def __init__(self, name: str, null_baseline: bool = False):
        self._name = name
        self._null_baseline = null_baseline

    @property
    def null_baseline(self) -> bool:
        """If True, prompts use a minimal classifier baseline (no HSC / jailbreak framing)."""
        return self._null_baseline

    @staticmethod
    def load_definition(definition_spec: dict[str, Any], domain: Optional[Domain] = None) -> "HateSpeechDefinition":
        """
        Factory method. ``definition_spec`` may be either:
          - a dict containing a ``path`` key pointing to a JSON file with the full config
          - a dict containing the full config inline (e.g. for vanilla definitions)
        ``domain`` is required for criteria/mixed definitions.
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
            if "null_baseline" in definition_spec:
                definition_config["null_baseline"] = definition_spec["null_baseline"]
            return VanillaHateSpeechDefinition._load_definition(definition_config)
        if definition_type == CriteriaHateSpeechDefinition.TYPE():
            include_definition_text = definition_spec.get("include_definition_text", False)
            return CriteriaHateSpeechDefinition._load_definition(definition_config, domain=domain, include_definition_text=include_definition_text)
        raise ValueError(
            f"Invalid hate speech definition type: {definition_type!r}. Expected one of: "
            f"{VanillaHateSpeechDefinition.TYPE()!r}, "
            f"{CriteriaHateSpeechDefinition.TYPE()!r}, "
        )

    @staticmethod
    @abstractmethod
    def _load_definition(definition_config: dict, **kwargs) -> "HateSpeechDefinition":
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
    """A plain-text hate speech definition."""

    def __init__(self, name: str, definition_text: str, null_baseline: bool = False):
        super().__init__(name, null_baseline=null_baseline)
        self._definition_text = definition_text

    @staticmethod
    def _load_definition(definition_config: dict, **kwargs) -> "VanillaHateSpeechDefinition":
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
        null_baseline = bool(definition_config.get("null_baseline", False))
        return VanillaHateSpeechDefinition(name, definition_text, null_baseline=null_baseline)

    def prompt_text(self) -> str:
        if self._definition_text:
            return "Reference plain-text definition: " + self._definition_text
        return ""

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
        domain: Domain,
        aspects: dict,
        definition_text: Optional[str] = None,
    ):
        super().__init__(name, null_baseline=False)
        self._domain = domain
        self._aspects = aspects
        self._definition_text = definition_text

    @staticmethod
    def _load_definition(
        definition_config: dict,
        domain: Optional[Domain] = None,
        include_definition_text: bool = False,
        **kwargs,
    ) -> "CriteriaHateSpeechDefinition":
        if domain is None:
            raise ValueError(
                "CriteriaHateSpeechDefinition requires a Domain instance. "
                "Provide one via HateSpeechDefinition.load_definition(..., domain=Domain.load(path))."
            )
        try:
            name = definition_config["name"]
            aspects = definition_config["aspects"]
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(
                f"Missing required key '{missing_key}' in CriteriaHateSpeechDefinition configuration. "
                "Expected keys: ['name', 'aspects'].\n"
                f"Offending config: {definition_config}"
            ) from e

        errors = domain.validate(aspects)
        if errors:
            bullet_list = "\n  - ".join(errors)
            raise ValueError(
                f"Definition '{name}' is not consistent with the domain:\n  - {bullet_list}"
            )

        definition_text = definition_config.get("definition_text") if include_definition_text else None
        return CriteriaHateSpeechDefinition(
            name=name,
            domain=domain,
            aspects=aspects,
            definition_text=definition_text,
        )

    def _resolve_selected(self, path: Tuple[str, ...]):
        node = self._aspects
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return Domain.extract_selected(node)

    @staticmethod
    def _format_value(selected) -> str:
        if selected is None:
            return Domain.UNSPECIFIED_LABEL
        if isinstance(selected, list):
            if not selected:
                return Domain.UNSPECIFIED_LABEL
            return ", ".join(str(v) for v in selected)
        if isinstance(selected, bool):
            return "yes" if selected else "no"
        return str(selected)

    def prompt_text(self) -> str:
        lines: List[str] = [f"Hate speech is defined using {'jointly the reference plain-text definition and ' if self._definition_text else ''}the extracted Hate Speech Criteria Aspects."]
        if self._definition_text:
            lines.append(f"Reference plain-text definition: {self._definition_text}")
        lines.append("Extracted Hate Speech Criteria Aspects:")
        for idx, (path, leaf) in enumerate(self._domain.iter_leaves(), start=1):
            description = (leaf.get("description") or "").strip()
            selected = self._resolve_selected(path)
            value_str = self._format_value(selected)
            label = " / ".join(p.replace("_", " ") for p in path)
            suffix = f" ({description})" if description else ""
            lines.append(f"{idx}. {label}{suffix}: {value_str}")
        return "\n".join(lines)

    @staticmethod
    def TYPE() -> str:
        return "criteria" 