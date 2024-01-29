from gpuhunt.providers.vastai import kilo, normalize_gpu_memory


class TestGPU:
    def test_normalize_known(self):
        assert normalize_gpu_memory("A100", 78 * kilo) == 80

    def test_normalize_unknown(self):
        assert normalize_gpu_memory("X1000", 78 * kilo + 10) == 78
