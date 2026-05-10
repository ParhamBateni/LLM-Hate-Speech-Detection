

from abc import abstractmethod

from typing import Optional, List, Tuple
from enum import Enum
from definitions import HateSpeechDefinition
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

class Prompting(ABC):
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
    def build_system_prompt(self, definition: Optional[HateSpeechDefinition], **kwargs) -> str:
        """
        Build a zero-shot system prompt for the given definition.
        """
        system_prompt = f"""
        You are a content moderation assistant. Your task is to classify input text as either HATE SPEECH or NOT HATE SPEECH{" based on HATE SPEECH DEFINITION" if definition else ""}.
        Your answer should be in the following format:
            PREDICTION: either 'hateful' or 'non-hateful'
            REASON: only one line explanation for your prediction
            
        HATE SPEECH DEFINITION: {definition.prompt_text() if definition else ""}
        """.strip()

        return system_prompt

    @property
    def name(self) -> str:
        """
        Returns the name of the prompting method.
        """
        return "zero-shot"


class FewShotPrompting(Prompting):
    class FewShotMode(Enum):
        RANDOM = "random"
        SMART = "smart"

    def __init__(self, num_shots: int, few_shot_mode: FewShotMode):
        self._num_shots = num_shots
        self._few_shot_mode = few_shot_mode

    def build_system_prompt(self, definition: Optional[HateSpeechDefinition], examples: List[Tuple[str, str]], **kwargs) -> str:
        """
        Build a few-shot system prompt for the given definition and examples.
        """
        system_prompt = f"""
        You are a content moderation assistant. Your task is to classify input text as either HATE SPEECH or NOT HATE SPEECH{" based on HATE SPEECH DEFINITION" if definition else ""}.
        Your answer should be in the following format:
            PREDICTION: either 'hateful' or 'non-hateful'
            REASON: only one line explanation for your prediction
            
        HATE SPEECH DEFINITION: {definition.prompt_text() if definition else ""}
        """
        for example in examples:
            system_prompt += f"""
            EXAMPLE:
            TEXT: {example[0]}
            PREDICTION: {example[1]}
            REASON: {example[2]}
            """
        return system_prompt

    @property
    def name(self) -> str:
        """
        Returns the name of the prompting method.
        """
        return "few-shot"

class ChainOfThoughtPrompting(Prompting):
    def build_system_prompt(self, definition: Optional[HateSpeechDefinition], **kwargs) -> str:
        """
        Build a chain-of-thought system prompt for the given definition.
        """
        system_prompt = f"""
        You are a content moderation assistant. Your task is to classify input text as either HATE SPEECH or NOT HATE SPEECH{" based on HATE SPEECH DEFINITION" if definition else ""}.
        Your answer should be in the following format:
            PREDICTION: either 'hateful' or 'non-hateful'
            REASON: only one line explanation for your prediction
            
        HATE SPEECH DEFINITION: {definition.prompt_text() if definition else ""}
        """
        # for example in examples:
        #     system_prompt += f"""
        #     EXAMPLE:
        #     TEXT: {example[0]}
        #     PREDICTION: {example[1]}
        #     REASON: {example[2]}
        #     """
        return system_prompt

    @property
    def name(self) -> str:
        """
        Returns the name of the prompting method.
        """
        return "chain-of-thought"
