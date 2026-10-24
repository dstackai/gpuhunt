import pytest

from gpuhunt.providers.aws import _get_gpu_memory_gib


class TestGetGpuMemoryGib:
    @pytest.mark.parametrize(
        ("name", "memory_mib", "expected"),
        [
            ("B300", 275040, 270),
            ("B200", 183359, 180),
            ("A10G", 22888, 24),
            ("L4", 22888, 24),
            ("L40S", 45776, 48),
            ("RTX PRO Server 6000", 98304, 96),
            ("A100", 40960, 40),
            ("A100", 81920, 80),
            ("V100", 16384, 16),
            ("V100", 32768, 32),
            ("A100", 39936, 40),
            ("A100", 81000, 80),
            ("A100", 61440, 60),
            ("unknown", 12345, 12345 / 1024),
        ],
    )
    def test_full_gpu_configuration(self, name, memory_mib, expected):
        assert _get_gpu_memory_gib(name, memory_mib) == expected
