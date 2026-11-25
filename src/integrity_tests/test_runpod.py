import re

import pytest

from gpuhunt import CatalogItem
from gpuhunt.providers.runpod import RunpodProvider
from integrity_tests.base import OffersIntegrityTests


class TestRunpodOffers(OffersIntegrityTests):
    @pytest.fixture(scope="class")
    def offers(self) -> list[CatalogItem]:
        return RunpodProvider.from_env().get()

    def test_secure_cloud_locations(self, offers: list[CatalogItem]) -> None:
        # Secure Cloud locations are data center IDs, e.g., EU-RO-1
        for offer in offers:
            assert re.fullmatch(r"[A-Z]+-[A-Z]+-\d+", offer.location), str(offer)

    def test_gpu_present(self, offers: list[CatalogItem]) -> None:
        gpus = {o.gpu_name for o in offers if o.gpu_count}
        assert len(gpus) >= 3, gpus

    def test_not_spot(self, offers: list[CatalogItem]) -> None:
        assert not any(o.spot for o in offers)

    def test_cpu_offers(self, offers: list[CatalogItem]) -> None:
        cpu_offers = [o for o in offers if o.gpu_count == 0]
        assert cpu_offers
        for offer in cpu_offers:
            # dstack parses the flavor ID and vCPU count from the name
            match = re.fullmatch(
                r"(?P<flavor>.+)-(?P<vcpu>\d+)-(?P<memory>\d+)", offer.instance_name
            )
            assert match, str(offer)
            assert int(match.group("vcpu")) == offer.cpu, str(offer)
            # Pod creation only accepts power-of-two vCPU counts
            assert offer.cpu & (offer.cpu - 1) == 0, str(offer)
            # The maximum container disk size
            assert offer.disk_size, str(offer)
