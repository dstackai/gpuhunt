import pytest

from gpuhunt.providers.aws import _get_gpu_memory_gib


class TestGetGpuMemoryGib:
    @pytest.mark.parametrize(
        ("name", "memory_mib", "expected"),
        [
            ("B300", 275040, 270),
            ("L4", 22888, 24),
            ("RTX PRO Server 6000", 98304, 96),
        ],
    )
    def test_full_gpu_configuration(self, name, memory_mib, expected):
        assert _get_gpu_memory_gib(name, memory_mib) == expected
