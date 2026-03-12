from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    @abstractmethod
    def __call__(prompts: list[str]) -> list[str]:
        pass


