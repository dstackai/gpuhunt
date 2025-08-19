"""Integrity tests for DigitalOcean provider."""

import os

import pytest
import requests

from gpuhunt._internal.constraints import KNOWN_AMD_GPUS, KNOWN_NVIDIA_GPUS

skip_if_no_token = pytest.mark.skipif(
    "DIGITAL_OCEAN_TOKEN" not in os.environ, reason="DIGITAL_OCEAN_TOKEN not set"
)


def fetch_digitalocean_sizes(api_url: str, token: str) -> list[dict]:
    response = requests.get(
        f"{api_url}/v2/sizes",
        headers={"Authorization": f"Bearer {token}"},
        params={"per_page": 500},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["sizes"]


def validate_gpu_model_names(gpu_sizes: list[dict]) -> None:
    for size in gpu_sizes:
        gpu_info = size["gpu_info"]
        gpu_model = gpu_info["model"]

        # Extract GPU name using same logic as provider
        model_parts = gpu_model.split("_")
        if len(model_parts) >= 3:
            # Handle cases like "nvidia_rtx6000_ada" -> "RTX6000ADA"
            gpu_name = "".join(part.upper() for part in model_parts[1:])
        else:
            # Handle cases like "amd_mi300x" -> "MI300X"
            gpu_name = model_parts[1].upper()

        gpu_found_in_nvidia = any(gpu.name.upper() == gpu_name for gpu in KNOWN_NVIDIA_GPUS)
        gpu_found_in_amd = any(gpu.name.upper() == gpu_name for gpu in KNOWN_AMD_GPUS)

        assert gpu_found_in_nvidia or gpu_found_in_amd, f"GPU '{gpu_name}' not found in known GPUs"


@skip_if_no_token
def test_standard_cloud_gpu_model_names():
    token = os.environ["DIGITAL_OCEAN_TOKEN"]
    sizes = fetch_digitalocean_sizes("https://api.digitalocean.com", token)

    gpu_sizes = [size for size in sizes if size.get("gpu_info")]
    validate_gpu_model_names(gpu_sizes)


@skip_if_no_token
def test_amd_cloud_gpu_model_names():
    token = os.environ["DIGITAL_OCEAN_TOKEN"]
    sizes = fetch_digitalocean_sizes("https://api-amd.digitalocean.com", token)

    gpu_sizes = [size for size in sizes if size.get("gpu_info")]
    validate_gpu_model_names(gpu_sizes)
