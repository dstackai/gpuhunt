from abc import ABC, abstractmethod

from gpuhunt._models import InstanceOffer


class AbstractProvider(ABC):
    @abstractmethod
    def get(self) -> list[InstanceOffer]:
        pass

    @classmethod
    def filter(cls, offers: list[InstanceOffer]) -> list[InstanceOffer]:
        return offers
