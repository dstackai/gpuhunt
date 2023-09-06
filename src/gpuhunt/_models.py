from typing import Any, Optional

from pydantic import BaseModel, model_validator


class InstanceOffer(BaseModel):
    instance_name: str
    location: Optional[str] = None  # region or zone
    price: Optional[float] = None  # $ per hour
    cpu: Optional[int] = None
    memory: Optional[float] = None  # in GB
    gpu_count: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory: Optional[float] = None  # in GB
    spot: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def parse_empty_as_none(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if value == "":
                    data[key] = None
        return data
