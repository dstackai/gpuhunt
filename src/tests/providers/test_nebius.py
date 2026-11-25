from unittest.mock import MagicMock

import pytest
from nebius.api.nebius.compute.v1 import Preset, PresetResources

from gpuhunt._internal.constraints import find_accelerators
from gpuhunt.providers import nebius


def test_user_agent_prefix(mocker):
    sdk_cls = mocker.patch.object(nebius, "SDK")
    mocker.patch.object(nebius, "CalculatorServiceClient")
    mocker.patch.object(nebius, "get_sample_projects", return_value={})
    mocker.patch.object(nebius, "__version__", "1.2.3")
    credentials = MagicMock()

    nebius.NebiusProvider(credentials).get()

    sdk_cls.assert_called_once_with(
        credentials=credentials,
        user_agent_prefix="gpuhunt/1.2.3",
    )


class TestMakeOffer:
    @pytest.mark.parametrize("region", ["eu-west2", "uk-south1", "us-north1"])
    @pytest.mark.parametrize("gpu_count", [1, 8])
    def test_b300_cluster_fabrics(self, region: str, gpu_count: int):
        preset = Preset(
            name=f"{gpu_count}gpu",
            resources=PresetResources(
                vcpu_count=24 * gpu_count,
                memory_gibibytes=346 * gpu_count,
                gpu_count=gpu_count,
            ),
            allow_gpu_clustering=gpu_count == 8,
        )
        offer = nebius._make_offer(
            platform="gpu-b300-sxm",
            preset=preset,
            gpu=find_accelerators(names=["B300"])[0],
            region=region,
            spot=True,
            price=4.09 * gpu_count,
        )

        assert offer is not None
        assert offer.provider_data["fabrics"] == ([f"{region}-a"] if gpu_count == 8 else [])
