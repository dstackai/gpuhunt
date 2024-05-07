import pytest

from gpuhunt.providers.oci import get_gpu_name


@pytest.mark.parametrize(
    ("shape_name", "gpu_name"),
    [
        ("VM.GPU.A10.2", "A10"),
        ("BM.GPU.A100-v2.8", "A100"),
        ("BM.GPU4.8", "A100"),
        ("VM.GPU3.4", "V100"),
        ("VM.GPU2.1", "P100"),
        ("BM.GPU.H100.8", "H100"),
        ("VM.Standard2.8", None),
        ("VM.Notgpu.A10", None),
    ],
)
def test_get_gpu_name(shape_name, gpu_name):
    assert get_gpu_name(shape_name) == gpu_name
