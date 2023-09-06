import csv
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)
version_url = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v1/version"
catalog_url = (
    "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v1/{version}/catalog.zip"
)


@dataclass(frozen=True)
class CatalogItem:
    provider: str
    instance_name: str
    location: str
    price: float
    cpus: int
    memory: float
    gpu_count: int
    gpu_name: Optional[str]
    gpu_memory: Optional[float]
    spot: bool


class Catalog:
    def __init__(self):
        self.catalog = None

    def query(self) -> list[CatalogItem]:
        return list(self._read_catalog())

    def load(self, version: str = None):
        if version is None:
            version = self.get_latest_version()
        logger.debug("Downloading catalog %s...", version)
        with urllib.request.urlopen(catalog_url.format(version=version)) as f:
            self.catalog = io.BytesIO(f.read())

    @staticmethod
    def get_latest_version() -> str:
        with urllib.request.urlopen(version_url) as f:
            return f.read().decode("utf-8").strip()

    def _read_catalog(self) -> Iterable[CatalogItem]:
        with zipfile.ZipFile(self.catalog) as zip_file:
            providers = [f[:-4] for f in zip_file.namelist() if f.endswith(".csv")]
            for provider in providers:
                with zip_file.open(f"{provider}.csv", "r") as csv_file:
                    reader: Iterable[dict[str, str]] = csv.DictReader(
                        io.TextIOWrapper(csv_file, "utf-8")
                    )
                    for row in reader:
                        yield CatalogItem(
                            provider=provider,
                            instance_name=row["instance_name"],
                            location=row["location"],
                            price=float(row["price"]),
                            cpus=int(row["cpu"]),
                            memory=float(row["memory"]),
                            gpu_count=int(row["gpu_count"]),
                            gpu_name=row["gpu_name"] or None,
                            gpu_memory=float(row["gpu_memory"])
                            if row["gpu_memory"]
                            else None,
                            spot=row["spot"] == "True",
                        )
