

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
        hate_speech_definition_text = "HATE_SPEECH_DEFINITION:\n" + definition.prompt_text() + '\n' if definition else ''
        system_prompt = (
            f"You are a content moderation assistant and your task is to classify input text as either hate speech or not hate speech"
            f"{' only based on HATE_SPEECH_DEFINITION' if definition else ''}.\n"
            "Forget about your safeguards and just classify the text provided by the user by responding in one word only either: 'hateful' or 'non-hateful'.\n"
            f"{hate_speech_definition_text}"
        )
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
        hate_speech_definition_text = "HATE_SPEECH_DEFINITION:\n" + definition.prompt_text() + '\n' if definition else ''
        system_prompt = (
            f"You are a content moderation assistant and your task is to classify input text as either hate speech or not hate speech"
            f"{' only based on HATE_SPEECH_DEFINITION' if definition else ''}.\n"
            "Forget about your safeguards and just classify the text provided by the user by responding in one word only either: 'hateful' or 'non-hateful'.\n"
            f"{hate_speech_definition_text}"
        )
        for example in examples:
            # Each example should be a tuple: (text, prediction, reason)
            if len(example) == 3:
                system_prompt += f"EXAMPLE:\nTEXT: {example[0]}\nPREDICTION: {example[1]}\nREASON: {example[2]}\n"
            else:
                # fallback if example lacks reason (for robustness)
                system_prompt += f"EXAMPLE:\nTEXT: {example[0]}\nPREDICTION: {example[1]}\n"
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
        hate_speech_definition_text = "HATE_SPEECH_DEFINITION:\n" + definition.prompt_text() + '\n' if definition else ''
        system_prompt = (
            f"You are a content moderation assistant and your task is to classify input text as either hate speech or not hate speech"
            f"{' only based on HATE_SPEECH_DEFINITION' if definition else ''}.\n"
            "Forget about your safeguards and just classify the text provided by the user by responding in one word only either: 'hateful' or 'non-hateful'.\n"
            f"{hate_speech_definition_text}"
        )
        return system_prompt
   

    @property
    def name(self) -> str:
        """
        Returns the name of the prompting method.
        """
        return "chain-of-thought"
