from abc import ABC, abstractmethod

from dstack.pricing.models import InstanceOffer


class AbstractProvider(ABC):
    @abstractmethod
    def get(self) -> list[InstanceOffer]:
        pass
