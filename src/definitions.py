from abc import ABC, abstractmethod

class HateSpeechDefinition(ABC):
    """
    An abstract base class that acts as an interface for hate speech definitions.
    Child classes should implement the prompt_text method.
    """
    def __init__(self, name: str):
        self._name = name
        
    @staticmethod
    def from_config(config: dict) -> "HateSpeechDefinition":
        """
        Factory method for creating a HateSpeechDefinition object from a configuration dictionary.
        """
        if config["type"] == VanillaHateSpeechDefinition.TYPE():
            return VanillaHateSpeechDefinition._from_config(config)
        elif config["type"] == CriteriaHateSpeechDefinition.TYPE():
            return CriteriaHateSpeechDefinition._from_config(config)
        elif config["type"] == MixedHateSpeechDefinition.TYPE():
            return MixedHateSpeechDefinition._from_config(config)
        else:
            raise ValueError(
                f"Invalid hate speech definition type: {config['type']}. "
                f"Expected one of: {VanillaHateSpeechDefinition.TYPE()}, "
                f"{CriteriaHateSpeechDefinition.TYPE()}, {MixedHateSpeechDefinition.TYPE()}"
            )

    @abstractmethod
    def _from_config(config: dict) -> "HateSpeechDefinition":
        """
        Factory method for creating a HateSpeechDefinition object from a configuration dictionary.
        """
        pass

    @abstractmethod
    def prompt_text(self) -> str:
        """
        Returns the hate speech definition as a human-readable prompt string.
        """
        pass

    @staticmethod
    @property
    def TYPE() -> str:
        """
        Returns the type of the hate speech definition.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the name of the hate speech definition.
        """
        return self.TYPE() + " (" + self._name + ")"

class VanillaHateSpeechDefinition(HateSpeechDefinition):
    """
    A plain-text hate speech definition.
    """

    def __init__(self, name: str, definition: str):
        super().__init__(name)
        self._definition = definition

    @staticmethod
    def _from_config(config: dict) -> "VanillaHateSpeechDefinition":
        """
        Factory method for creating a VanillaHateSpeechDefinition object from a configuration dictionary.
        """
        try:
            return VanillaHateSpeechDefinition(config["name"], config["text"])
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(
                f"Missing required key '{missing_key}' in VanillaHateSpeechDefinition configuration dictionary. "
                "The expected keys are: ['definition']. "
                f"Offending config: {config}"
            ) from e

    def prompt_text(self) -> str:
        return self._definition

    @staticmethod
    def TYPE() -> str:
        return "vanilla"

class CriteriaHateSpeechDefinition(HateSpeechDefinition):
    """
    A hate speech definition specified using set criteria.
    """

    def __init__(
        self,
        name: str,
        target_group: str,
        dominance: str,
        perpetrator_characteristics: str,
        negative_reference: str,
        potential_consequences: str,
    ):
        super().__init__(name)
        self._target_group = target_group
        self._dominance = dominance
        self._perpetrator_characteristics = perpetrator_characteristics
        self._negative_reference = negative_reference
        self._potential_consequences = potential_consequences

    @staticmethod
    def _from_config(config: dict) -> "CriteriaHateSpeechDefinition":
        """
        Factory method for creating a CriteriaHateSpeechDefinition object from a configuration dictionary.
        """
        try:
            return CriteriaHateSpeechDefinition(config["name"], config["target_group"], config["dominance"], config["perpetrator_characteristics"], config["negative_reference"], config["potential_consequences"])
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(
                f"Missing required key '{missing_key}' in CriteriaHateSpeechDefinition configuration dictionary. "
                "The expected keys are: ['target_group', 'dominance', 'perpetrator_characteristics', 'negative_reference', 'potential_consequences']. "
                f"Offending config: {config}"
            ) from e

    def prompt_text(self) -> str:
        return ("Hate speech is defined using the following template based on5 criteria:\n"
            f"1- Target group: List of target groups that if targetted might be considered for hate speech." # TODO: investigate the effect of might and should
            f"2- Dominance: List of dominant groups to take into account as a potential target of hate speech."
            f"3- Perpetrator characteristics: List of characteristics of the perpetrator that should be taken into account e.g. socitial roles or being a member of the target group itself."
            f"4- Negative reference: List of negative references to the target group that should be taken into account. e.g. stereotypes, group characteristics, slurs, etc." # TODO: investigate the effect of adding explicit or implicit
            f"5- Potential consequences: List of potential consequences of the hate speech that should be taken into account. e.g. violence, hate, discrimination, etc."
            f"The hate speech definition is then defined as follows:\n"
            f"Target group: {self._target_group}\n"
            f"Dominance: {self._dominance}\n"
            f"Perpetrator characteristics: {self._perpetrator_characteristics}\n"
            f"Negative reference: {self._negative_reference}\n"
            f"Potential consequences: {self._potential_consequences}"
        )

    @staticmethod
    def TYPE() -> str:
        return "criteria"

class MixedHateSpeechDefinition(HateSpeechDefinition):
    """
    A hate speech definition that is a mix of plain-text and set criteria.
    """

    def __init__(self, name: str, plain_text_definition: str, set_criteria_definition: CriteriaHateSpeechDefinition):
        super().__init__(name)
        self._plain_text_definition = plain_text_definition
        self._set_criteria_definition = set_criteria_definition

    @staticmethod
    def _from_config(config: dict) -> "MixedHateSpeechDefinition":
        """
        Factory method for creating a MixedHateSpeechDefinition object from a configuration dictionary.
        """
        try:
            return MixedHateSpeechDefinition(config["name"], config["plain_text_definition"], CriteriaHateSpeechDefinition._from_config(config["set_criteria_definition"]))
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(
                f"Missing required key '{missing_key}' in MixedHateSpeechDefinition configuration dictionary. "
                "The expected keys are: ['plain_text_definition', 'set_criteria_definition']. "
                f"Offending config: {config}"
            ) from e
   
 

    def prompt_text(self) -> str:
        return f"Definition:\n{self._plain_text_definition}\n\nSet criteria:\n{self._set_criteria_definition.prompt_text()}"

    @staticmethod
    def TYPE() -> str:
        return "mixed"