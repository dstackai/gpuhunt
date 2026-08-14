import pytest

from gpuhunt._internal.constraints import KNOWN_AMD_GPUS
from gpuhunt._internal.models import AMDArchitecture


@pytest.mark.parametrize(
    ["model", "architecture", "expected_memory"],
    [
        pytest.param("MI325X", AMDArchitecture.CDNA3, 288, id="MI325X"),
        pytest.param("MI308X", AMDArchitecture.CDNA3, 128, id="MI308X"),
        pytest.param("MI300X", AMDArchitecture.CDNA3, 192, id="MI300X"),
        pytest.param("MI300A", AMDArchitecture.CDNA3, 128, id="MI300A"),
        pytest.param("MI250X", AMDArchitecture.CDNA2, 128, id="MI250X"),
        pytest.param("MI250", AMDArchitecture.CDNA2, 128, id="MI250"),
        pytest.param("MI210", AMDArchitecture.CDNA2, 64, id="MI210"),
        pytest.param("MI100", AMDArchitecture.CDNA, 32, id="MI100"),
    ],
)
def test_amd_gpu_architecture(model: str, architecture: AMDArchitecture, expected_memory: int):
    for gpu in KNOWN_AMD_GPUS:
        if gpu.name == model:
            assert gpu.architecture == architecture
            assert gpu.memory == expected_memory
            return
    # If we get here, the test should fail since we could not find the GPU in our known list.
    assert False
