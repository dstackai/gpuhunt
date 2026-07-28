import json

import pytest

from gpuhunt.providers.oci import CostEstimatorShapeList, get_gpu_name

# Trimmed excerpts of the real Cost Estimator payload
# (https://www.oracle.com/a/ocom/docs/cloudestimator2/data/shapes.json).
# The `disk` product of BM.DenseIO.E5.128 reports a fractional qty of 81.6.
SHAPES = {
    "items": [
        {
            "name": "BM.DenseIO.E5.128",
            "hidden": False,
            "status": "ACTIVE",
            "allowPreemptible": False,
            "bundleMemoryQty": 1536,
            "gpuQty": None,
            "gpuMemoryQty": None,
            "processorType": {"value": "amd"},
            "shapeType": {"value": "bm"},
            "subType": {"value": "dense"},
            "products": [
                {"partNumber": "B98202", "qty": 128, "type": {"value": "ocpu"}},
                {"partNumber": "B98203", "qty": 1536, "type": {"value": "memory"}},
                {"partNumber": "B98204", "qty": 81.6, "type": {"value": "disk"}},
            ],
        },
        {
            "name": "BM.Standard.E4.128",
            "hidden": False,
            "status": "ACTIVE",
            "allowPreemptible": False,
            "bundleMemoryQty": 2048,
            "gpuQty": None,
            "gpuMemoryQty": None,
            "processorType": {"value": "amd"},
            "shapeType": {"value": "bm"},
            "subType": {"value": "standard"},
            "products": [
                {"partNumber": "B93113", "qty": 128, "type": {"value": "ocpu"}},
                {"partNumber": "B93114", "qty": 2048, "type": {"value": "memory"}},
            ],
        },
    ]
}


class TestCostEstimatorShapeList:
    def test_fractional_qty_is_truncated(self):
        """Some integral quantities are reported as floats. Pydantic v1 truncated them, v2
        errors out, which would fail the whole document and not just the shape we do not
        even use."""
        shapes = CostEstimatorShapeList.model_validate_json(json.dumps(SHAPES))
        assert [product.qty for product in shapes.items[0].products] == [128, 1536, 81]
        assert shapes.items[1].name == "BM.Standard.E4.128"


class TestGetGpuName:
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
    def test_get_gpu_name(self, shape_name, gpu_name):
        assert get_gpu_name(shape_name) == gpu_name
