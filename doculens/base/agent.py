from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def invoke(self, payload: any) -> any:
        raise NotImplementedError

    def prettify(self, payload): ...
