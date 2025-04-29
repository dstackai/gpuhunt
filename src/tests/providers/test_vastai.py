import warnings

import requests

from gpuhunt.providers.vastai import get_dstack_gpu_name, get_vastai_gpu_names


def test_real_world_vastai_offers():
    """Test that GPU name conversion works correctly for all real-world offers from Vast.ai."""
    # Get all offers from Vast.ai
    response = requests.post(
        "https://cloud.vast.ai/api/v0/bundles/", json={"limit": 3000}, timeout=10
    )
    response.raise_for_status()
    offers = response.json()["offers"]

    # Track unique GPU names and their conversions
    unique_gpu_names = set()
    conversion_issues = set()

    for offer in offers:
        vastai_gpu_name = offer["gpu_name"]
        if not vastai_gpu_name:
            continue

        unique_gpu_names.add(vastai_gpu_name)

        # Convert to dstack format and back
        dstack_name = get_dstack_gpu_name(vastai_gpu_name)
        vastai_names = get_vastai_gpu_names(dstack_name)

        # Check if the original name is in the converted back list
        if vastai_gpu_name not in vastai_names:
            conversion_issues.add(vastai_gpu_name)

    # Print statistics about mismatched GPUs
    if conversion_issues:
        warning_msg = f"Found {len(conversion_issues)} GPU names without valid mapping:\n"
        for gpu_name in sorted(conversion_issues):
            warning_msg += f"- {gpu_name}\n"
        warnings.warn(warning_msg)
