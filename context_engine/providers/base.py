"""Abstract provider interface."""

from abc import ABC, abstractmethod


class Provider(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a single completion. Returns the text content."""
        ...
