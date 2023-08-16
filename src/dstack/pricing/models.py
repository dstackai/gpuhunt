from typing import Optional

from pydantic import BaseModel, field_validator


class InstanceOffer(BaseModel):
    instance_name: str
    location: str  # region or zone
    price: float  # $ per hour
    cpu: Optional[int] = None
    memory: Optional[float] = None  # in GB
    gpu_count: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory: Optional[float] = None  # in GB
    spot: bool

    @field_validator("cpu", "memory", "gpu_count", "gpu_name", "gpu_memory", mode="before")
    @classmethod
    def parse_empty_as_none(cls, v: str):
        if v == "":
            return None
        return v
