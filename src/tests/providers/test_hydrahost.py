import copy

import pytest
import requests

from gpuhunt._internal.constraints import matches
from gpuhunt._internal.models import AcceleratorVendor, CPUArchitecture, QueryFilter
from gpuhunt.providers.hydrahost import (
    INVENTORY_URL,
    HydraHostProvider,
    get_cpu_arch,
    get_cpu_count,
    get_hourly_price,
    get_location,
    get_root_disk_gib,
)

L40S_ID = "6fc5bebd-524c-4b4d-b096-af92c3d7bf44"
CPU_ID = "01d7db09-0b3f-46c5-b00f-5722091314c2"
A6000_ID = "ade0c92b-ea0f-4eb6-9c69-13d08ff09d3f"

# Recorded from https://brokkr.hydrahost.com/api/v1/inventory?filters=status:eq:on%20demand
# on 2026-11-09 (three of the five listings; OS lists and price periods trimmed).
RECORDED_PAGE = {
    "data": [
        {
            "id": "6fc5bebd-524c-4b4d-b096-af92c3d7bf44",
            "name": "rackscentral-h-08c0eb6aa1cb",
            "location": "Singapore",
            "role": "Server",
            "stockStatus": "on demand",
            "isTeeCapable": False,
            "networking": {"networkType": "Public", "vpcCapable": False},
            "specs": {
                "cpu": {
                    "coresPerCpu": 64,
                    "count": 2,
                    "model": "AMD EPYC 7763 64-Core Processor",
                    "threadsPerCore": 2,
                    "threadsPerCpu": 128,
                    "totalCores": 128,
                    "totalThreads": 256,
                },
                "gpu": {"count": 5, "model": "NVIDIA L40S"},
                "memory": {"total": 2048},
                "storage": {
                    "hddCount": 0,
                    "hddSize": 0,
                    "nvmeCount": 4,
                    "nvmeSize": 28616,
                    "ssdCount": 1,
                    "ssdSize": 447,
                    "total": 29063,
                },
            },
            "listing": {
                "isInterruptibleOnly": False,
                "isActive": True,
                "onDemandPrice": {"perHour": {"perGpu": 95, "perCpu": 237.5, "total": 475}},
                "interruptiblePrice": {"perHour": {"perGpu": 68, "perCpu": 170, "total": 340}},
            },
            "availableOperatingSystems": [
                {
                    "id": "60d2ab93-d50d-4ec8-9cbd-b8a30b0130f3",
                    "name": "Debian 12 (Bookworm) — HPC",
                    "slug": "debian-bookworm-hpc",
                },
                {
                    "id": "a8215674-d1f5-461a-94f6-f123001155a2",
                    "name": "Debian 12 (Bookworm) — VANILLA",
                    "slug": "debian-bookworm-vanilla",
                },
            ],
            "storageLayouts": {
                "configs": [
                    {
                        "disks": [{"wwn": "0x538f6016200ad94e", "name": "sda"}],
                        "disk_type": "ssd",
                        "capabilities": ["direct", "lvm"],
                        "num_disks": 1,
                        "size_per_disk": 480103981056,
                        "disk_group_name": "SSD_447GB",
                        "file_systems": ["ext4", "xfs"],
                    },
                    {
                        "disks": [
                            {"wwn": "eui.01000000000000005cd2e48460715351", "name": "nvme0n1"},
                            {"wwn": "eui.01000000000000005cd2e4516c755351", "name": "nvme1n1"},
                            {"wwn": "eui.01000000000000005cd2e48660715351", "name": "nvme2n1"},
                            {"wwn": "eui.01000000000000005cd2e43960715351", "name": "nvme3n1"},
                        ],
                        "disk_type": "nvme",
                        "capabilities": [
                            "direct",
                            "lvm",
                            "raid0",
                            "raid1",
                            "raid10",
                            "raid5",
                            "raid6",
                        ],
                        "num_disks": 4,
                        "size_per_disk": 7681501126656,
                        "disk_group_name": "NVME_7153GB",
                        "file_systems": ["xfs"],
                    },
                ],
                "default": {
                    "os_disks_group": {
                        "config": "lvm",
                        "file_system": "ext4",
                        "group": "SSD_447GB",
                        "mountpoint": "/",
                    },
                    "data_disks_groups": [
                        {
                            "config": "lvm",
                            "file_system": "ext4",
                            "group": "NVME_7153GB",
                            "mountpoint": "/data0",
                        }
                    ],
                    "cold_storage_disks_groups": [],
                },
            },
            "defaultDiskLayouts": [
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/",
                    "diskType": "ssd",
                    "disks": ["0x538f6016200ad94e"],
                },
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/data0",
                    "diskType": "nvme",
                    "disks": [
                        "eui.01000000000000005cd2e48460715351",
                        "eui.01000000000000005cd2e4516c755351",
                        "eui.01000000000000005cd2e48660715351",
                        "eui.01000000000000005cd2e43960715351",
                    ],
                },
            ],
            "isInterruptibleDeployment": False,
            "interruptibleNoticePeriod": None,
        },
        {
            "id": "01d7db09-0b3f-46c5-b00f-5722091314c2",
            "name": "rackscentral-h-84160c3501a9",
            "location": "Singapore",
            "role": "Server",
            "stockStatus": "on demand",
            "isTeeCapable": False,
            "networking": {"networkType": "Public", "vpcCapable": False},
            "specs": {
                "cpu": {
                    "coresPerCpu": 64,
                    "count": 2,
                    "model": "AMD EPYC 7763 64-Core Processor",
                    "threadsPerCore": 2,
                    "threadsPerCpu": 128,
                    "totalCores": 128,
                    "totalThreads": 256,
                },
                "gpu": {"count": None, "model": None},
                "memory": {"total": 2036},
                "storage": {
                    "hddCount": 0,
                    "hddSize": 0,
                    "nvmeCount": 4,
                    "nvmeSize": 28614,
                    "ssdCount": 1,
                    "ssdSize": 447,
                    "total": 29061,
                },
            },
            "listing": {
                "isInterruptibleOnly": False,
                "isActive": True,
                "onDemandPrice": {"perHour": {"perGpu": None, "perCpu": 130, "total": 260}},
                "interruptiblePrice": {"perHour": {"perGpu": None, "perCpu": None, "total": None}},
            },
            "availableOperatingSystems": [
                {
                    "id": "a8215674-d1f5-461a-94f6-f123001155a2",
                    "name": "Debian 12 (Bookworm) — VANILLA",
                    "slug": "debian-bookworm-vanilla",
                },
                {
                    "id": "c85a6593-c182-489a-8635-8463286986d1",
                    "name": "Ubuntu 22.04 (Jammy) — VANILLA",
                    "slug": "ubuntu-jammy-vanilla",
                },
            ],
            "storageLayouts": {
                "configs": [
                    {
                        "disks": [{"wwn": "0x538f6016200add6b", "name": "sda"}],
                        "disk_type": "ssd",
                        "capabilities": ["direct", "lvm"],
                        "num_disks": 1,
                        "size_per_disk": 480103981056,
                        "disk_group_name": "SSD_480GB",
                        "file_systems": ["ext4", "xfs"],
                    },
                    {
                        "disks": [
                            {"wwn": "eui.01000000000000005cd2e42697755351", "name": "nvme1n1"},
                            {"wwn": "eui.01000000000000005cd2e4631d715351", "name": "nvme2n1"},
                            {"wwn": "eui.01000000000000005cd2e4f05f715351", "name": "nvme0n1"},
                            {"wwn": "eui.01000000000000005cd2e4f750715351", "name": "nvme3n1"},
                        ],
                        "disk_type": "nvme",
                        "capabilities": ["lvm", "raid0", "raid1", "raid10", "raid5", "raid6"],
                        "num_disks": 4,
                        "size_per_disk": 7681501126656,
                        "disk_group_name": "NVME_7681GB",
                        "file_systems": ["xfs"],
                    },
                ],
                "default": {
                    "os_disks_group": {
                        "config": "lvm",
                        "file_system": "ext4",
                        "group": "SSD_480GB",
                        "mountpoint": "/",
                    },
                    "data_disks_groups": [
                        {
                            "config": "lvm",
                            "file_system": "ext4",
                            "group": "NVME_7681GB",
                            "mountpoint": "/data0",
                        }
                    ],
                    "cold_storage_disks_groups": [],
                },
            },
            "defaultDiskLayouts": [
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/",
                    "diskType": "ssd",
                    "disks": ["0x538f6016200add6b"],
                },
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/data0",
                    "diskType": "nvme",
                    "disks": [
                        "eui.01000000000000005cd2e42697755351",
                        "eui.01000000000000005cd2e4631d715351",
                        "eui.01000000000000005cd2e4f05f715351",
                        "eui.01000000000000005cd2e4f750715351",
                    ],
                },
            ],
            "isInterruptibleDeployment": False,
            "interruptibleNoticePeriod": None,
        },
        {
            "id": "ade0c92b-ea0f-4eb6-9c69-13d08ff09d3f",
            "name": "massedcompute-h-ac1f6b2505b8",
            "location": "Missouri",
            "role": "Server",
            "stockStatus": "on demand",
            "isTeeCapable": False,
            "networking": {"networkType": "NAT", "vpcCapable": False},
            "specs": {
                "cpu": {
                    "coresPerCpu": 22,
                    "count": 2,
                    "model": "Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz",
                    "threadsPerCore": 2,
                    "threadsPerCpu": 44,
                    "totalCores": 44,
                    "totalThreads": 88,
                },
                "gpu": {"count": 8, "model": "NVIDIA RTX A6000"},
                "memory": {"total": 512},
                "storage": {
                    "hddCount": 0,
                    "hddSize": 0,
                    "nvmeCount": 0,
                    "nvmeSize": 0,
                    "ssdCount": 8,
                    "ssdSize": 32119,
                    "total": 32119,
                },
            },
            "listing": {
                "isInterruptibleOnly": False,
                "isActive": True,
                "onDemandPrice": {"perHour": {"perGpu": 60, "perCpu": 240, "total": 480}},
                "interruptiblePrice": {"perHour": {"perGpu": None, "perCpu": None, "total": None}},
            },
            "availableOperatingSystems": [
                {
                    "id": "60d2ab93-d50d-4ec8-9cbd-b8a30b0130f3",
                    "name": "Debian 12 (Bookworm) — HPC",
                    "slug": "debian-bookworm-hpc",
                },
                {
                    "id": "a8215674-d1f5-461a-94f6-f123001155a2",
                    "name": "Debian 12 (Bookworm) — VANILLA",
                    "slug": "debian-bookworm-vanilla",
                },
            ],
            "storageLayouts": {
                "configs": [
                    {
                        "disks": [{"wwn": "0x5002538c403b7a2e", "name": "sda"}],
                        "disk_type": "ssd",
                        "capabilities": ["direct", "lvm"],
                        "num_disks": 1,
                        "size_per_disk": 480103981056,
                        "disk_group_name": "SSD_480GB",
                        "file_systems": ["ext4", "xfs"],
                    },
                    {
                        "disks": [
                            {"wwn": "0x5002538f31a294f2", "name": "sdb"},
                            {"wwn": "0x5002538f329024f3", "name": "sdc"},
                            {"wwn": "0x5002538f32808e9d", "name": "sdd"},
                            {"wwn": "0x5002538f54a13133", "name": "sdg"},
                        ],
                        "disk_type": "ssd",
                        "capabilities": ["lvm", "raid0", "raid1", "raid10", "raid5", "raid6"],
                        "num_disks": 4,
                        "size_per_disk": 4000787030016,
                        "disk_group_name": "SSD_4000GB",
                        "file_systems": ["ext4", "xfs"],
                    },
                    {
                        "disks": [{"wwn": "0x5002538f34313026", "name": "sde"}],
                        "disk_type": "ssd",
                        "capabilities": ["direct", "lvm"],
                        "num_disks": 1,
                        "size_per_disk": 2000398934016,
                        "disk_group_name": "SSD_2000GB",
                        "file_systems": ["ext4", "xfs"],
                    },
                    {
                        "disks": [
                            {"wwn": "0x5002538f34909621", "name": "sdf"},
                            {"wwn": "0x5002538f3490949c", "name": "sdh"},
                        ],
                        "disk_type": "ssd",
                        "capabilities": ["lvm", "raid0", "raid1"],
                        "num_disks": 2,
                        "size_per_disk": 8001563222016,
                        "disk_group_name": "SSD_8001GB",
                        "file_systems": ["xfs"],
                    },
                ],
                "default": {
                    "os_disks_group": {
                        "config": "lvm",
                        "file_system": "ext4",
                        "group": "SSD_480GB",
                        "mountpoint": "/",
                    },
                    "data_disks_groups": [
                        {
                            "config": "lvm",
                            "file_system": "ext4",
                            "group": "SSD_2000GB",
                            "mountpoint": "/data0",
                        },
                        {
                            "config": "lvm",
                            "file_system": "ext4",
                            "group": "SSD_4000GB",
                            "mountpoint": "/data1",
                        },
                        {
                            "config": "lvm",
                            "file_system": "ext4",
                            "group": "SSD_8001GB",
                            "mountpoint": "/data2",
                        },
                    ],
                    "cold_storage_disks_groups": [],
                },
            },
            "defaultDiskLayouts": [
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/",
                    "diskType": "ssd",
                    "disks": ["0x5002538c403b7a2e"],
                },
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/data1",
                    "diskType": "ssd",
                    "disks": [
                        "0x5002538f31a294f2",
                        "0x5002538f329024f3",
                        "0x5002538f32808e9d",
                        "0x5002538f54a13133",
                    ],
                },
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/data0",
                    "diskType": "ssd",
                    "disks": ["0x5002538f34313026"],
                },
                {
                    "config": "lvm",
                    "format": "ext4",
                    "mountpoint": "/data2",
                    "diskType": "ssd",
                    "disks": ["0x5002538f34909621", "0x5002538f3490949c"],
                },
            ],
            "isInterruptibleDeployment": False,
            "interruptibleNoticePeriod": None,
        },
    ],
    "meta": {"page": 1, "pageSize": 100, "totalItems": 3, "totalPages": 1},
}


def page_of(*devices: dict, page: int = 1, total_pages: int = 1) -> dict:
    return {
        "data": list(devices),
        "meta": {
            "page": page,
            "pageSize": 100,
            "totalItems": len(devices),
            "totalPages": total_pages,
        },
    }


def device(device_id: str) -> dict:
    return copy.deepcopy(next(d for d in RECORDED_PAGE["data"] if d["id"] == device_id))


def mock_inventory(requests_mock, on_demand: dict, interruptible: dict | None = None) -> None:
    """Serve `on_demand` for the on-demand query and `interruptible` for the interruptible one."""
    interruptible = interruptible if interruptible is not None else page_of()

    def respond(request, context):
        if "interruptibleready:eq:true" in request.qs["filters"][0]:
            return interruptible
        return on_demand

    requests_mock.get(INVENTORY_URL, json=respond)


@pytest.fixture
def recorded(requests_mock):
    mock_inventory(
        requests_mock,
        on_demand=copy.deepcopy(RECORDED_PAGE),
        interruptible=page_of(device(L40S_ID)),
    )


class TestGet:
    def test_offers_from_recorded_page(self, recorded):
        offers = HydraHostProvider().get()

        # L40S server: on-demand + interruptible; CPU server and A6000 server: on-demand only
        assert sorted((o.instance_name, o.spot) for o in offers) == sorted(
            [(L40S_ID, False), (L40S_ID, True), (CPU_ID, False), (A6000_ID, False)]
        )
        prices = [o.price for o in offers]
        assert prices == sorted(prices)

    def test_whole_server_offer(self, recorded):
        offers = HydraHostProvider().get()

        offer = next(o for o in offers if o.instance_name == L40S_ID and not o.spot)
        assert offer.provider == "hydrahost"
        assert offer.location == "singapore"
        assert offer.price == 4.75  # perHour.total is in cents
        assert offer.cpu == 256  # 2 x 64 cores x 2 threads
        assert offer.memory == 2048
        assert offer.gpu_vendor == AcceleratorVendor.NVIDIA
        assert offer.gpu_count == 5
        assert offer.gpu_name == "L40S"
        assert offer.gpu_memory == 48
        # The 480 GB OS disk mounted at /, not the 29 TB total that includes /data0
        assert offer.disk_size == 447
        assert offer.cpu_arch == CPUArchitecture.X86
        assert offer.flags == []
        assert offer.provider_data == {
            "device_name": "rackscentral-h-08c0eb6aa1cb",
            "operating_systems": ["debian-bookworm-hpc", "debian-bookworm-vanilla"],
            "tee_capable": False,
            "network_type": "Public",
        }

    def test_interruptible_offer(self, recorded):
        offers = HydraHostProvider().get()

        spot = [o for o in offers if o.spot]
        assert len(spot) == 1
        assert spot[0].instance_name == L40S_ID
        assert spot[0].price == 3.4
        assert spot[0].gpu_count == 5

    def test_cpu_only_server(self, recorded):
        offers = HydraHostProvider().get()

        offer = next(o for o in offers if o.instance_name == CPU_ID)
        assert offer.price == 2.6
        assert offer.gpu_count == 0
        assert offer.gpu_name is None
        assert offer.gpu_memory is None
        assert offer.gpu_vendor is None
        assert offer.cpu == 256

    def test_nat_server(self, recorded):
        offers = HydraHostProvider().get()

        offer = next(o for o in offers if o.instance_name == A6000_ID)
        assert offer.location == "missouri"
        assert offer.price == 4.8
        assert offer.cpu == 88
        assert offer.memory == 512
        assert offer.gpu_name == "A6000"
        assert offer.gpu_count == 8
        assert offer.provider_data["network_type"] == "NAT"

    def test_no_spot_offer_unless_interruptible_ready(self, requests_mock):
        # The listing carries an interruptible price, but the server is not interruptible-ready.
        mock_inventory(requests_mock, on_demand=page_of(device(L40S_ID)))

        offers = HydraHostProvider().get()

        assert [o.spot for o in offers] == [False]

    def test_interruptible_only_listing(self, requests_mock):
        d = device(L40S_ID)
        d["listing"]["isInterruptibleOnly"] = True
        d["interruptibleNoticePeriod"] = 3600000
        mock_inventory(requests_mock, on_demand=page_of(d), interruptible=page_of(d))

        offers = HydraHostProvider().get()

        assert [o.spot for o in offers] == [True]
        assert offers[0].provider_data["interruptible_notice_period_ms"] == 3600000

    def test_inactive_listing_skipped(self, requests_mock):
        d = device(A6000_ID)
        d["listing"]["isActive"] = False
        mock_inventory(requests_mock, on_demand=page_of(d))

        assert HydraHostProvider().get() == []

    def test_unknown_gpu_skipped(self, requests_mock, caplog):
        d = device(A6000_ID)
        d["specs"]["gpu"]["model"] = "NVIDIA Future GPU"
        mock_inventory(requests_mock, on_demand=page_of(d, device(CPU_ID)))

        offers = HydraHostProvider().get()

        assert [o.instance_name for o in offers] == [CPU_ID]
        assert "unknown GPU 'NVIDIA Future GPU'" in caplog.text

    def test_listing_without_root_disk_skipped(self, requests_mock, caplog):
        d = device(A6000_ID)
        d["defaultDiskLayouts"] = [x for x in d["defaultDiskLayouts"] if x["mountpoint"] != "/"]
        mock_inventory(requests_mock, on_demand=page_of(d))

        assert HydraHostProvider().get() == []
        assert "root disk size unknown" in caplog.text

    def test_listing_without_price_skipped(self, requests_mock):
        d = device(A6000_ID)
        d["listing"]["onDemandPrice"]["perHour"]["total"] = None
        mock_inventory(requests_mock, on_demand=page_of(d))

        assert HydraHostProvider().get() == []

    def test_follows_pages(self, requests_mock):
        pages = {
            "1": page_of(device(L40S_ID), page=1, total_pages=2),
            "2": page_of(device(A6000_ID), page=2, total_pages=2),
        }

        def respond(request, context):
            if "interruptibleready:eq:true" in request.qs["filters"][0]:
                return page_of()
            return pages[request.qs["page"][0]]

        requests_mock.get(INVENTORY_URL, json=respond)

        offers = HydraHostProvider().get()

        assert {o.instance_name for o in offers} == {L40S_ID, A6000_ID}
        on_demand_calls = [
            r for r in requests_mock.request_history if "interruptible" not in r.qs["filters"][0]
        ]
        assert [r.qs["page"] for r in on_demand_calls] == [["1"], ["2"]]
        assert all(r.qs["pagesize"] == ["100"] for r in on_demand_calls)

    def test_interruptible_query_failure_keeps_on_demand_offers(self, requests_mock, caplog):
        def respond(request, context):
            if "interruptibleready:eq:true" in request.qs["filters"][0]:
                context.status_code = 500
                return {"message": "boom"}
            return page_of(device(L40S_ID))

        requests_mock.get(INVENTORY_URL, json=respond)

        offers = HydraHostProvider().get()

        assert [o.spot for o in offers] == [False]
        assert "Failed to fetch interruptible Hydra Host listings" in caplog.text

    def test_on_demand_query_failure_raises(self, requests_mock):
        requests_mock.get(INVENTORY_URL, status_code=503)

        with pytest.raises(requests.HTTPError):
            HydraHostProvider().get()

    def test_unexpected_response_raises(self, requests_mock):
        requests_mock.get(INVENTORY_URL, json=[])

        with pytest.raises(ValueError):
            HydraHostProvider().get()

    def test_query_filter(self, recorded):
        offers = HydraHostProvider().get()

        q = QueryFilter(gpu_name=["L40S"], spot=False)
        assert [o.instance_name for o in offers if matches(o, q)] == [L40S_ID]
        q = QueryFilter(min_disk_size=1000)
        assert [o for o in offers if matches(o, q)] == []

    def test_from_env_needs_no_credentials(self):
        assert isinstance(HydraHostProvider.from_env(), HydraHostProvider)


class TestHelpers:
    def test_hourly_price_is_in_dollars(self):
        assert get_hourly_price({"perHour": {"total": 475}}) == 4.75
        assert get_hourly_price({"perHour": {"total": None}}) is None
        assert get_hourly_price({"perHour": {"total": 0}}) is None
        assert get_hourly_price(None) is None

    def test_cpu_count(self):
        assert get_cpu_count({"totalCores": 44, "threadsPerCore": 2, "totalThreads": 88}) == 88
        assert get_cpu_count({"totalCores": 44, "threadsPerCore": 2}) == 88
        assert get_cpu_count({"totalCores": 64}) == 64
        assert get_cpu_count({}) == 0

    def test_cpu_arch(self):
        assert get_cpu_arch("AMD EPYC 7763 64-Core Processor", "L40S") == CPUArchitecture.X86
        assert get_cpu_arch("Neoverse-V2", None) == CPUArchitecture.ARM
        assert get_cpu_arch(None, "GH200") == CPUArchitecture.ARM

    def test_location(self):
        assert get_location("Singapore") == "singapore"
        assert get_location("Kansas City, MO") == "kansas-city-mo"
        assert get_location(None) == ""

    @pytest.mark.parametrize(
        ("config", "expected"),
        [("lvm", 2 * 3726), ("raid0", 2 * 3726), ("raid1", 3726)],
    )
    def test_root_disk_spanning_disks(self, config, expected):
        size = 4000787030016  # 4 TB in bytes, 3726 GiB
        d = {
            "defaultDiskLayouts": [
                {"config": config, "mountpoint": "/", "disks": ["a", "b"]},
                {"config": "lvm", "mountpoint": "/data0", "disks": ["c"]},
            ],
            "storageLayouts": {
                "configs": [
                    {"size_per_disk": size, "disks": [{"wwn": "a"}, {"wwn": "b"}]},
                    {"size_per_disk": 10 * size, "disks": [{"wwn": "c"}]},
                ]
            },
        }
        assert get_root_disk_gib(d) == expected

    def test_root_disk_unknown(self):
        d = {"defaultDiskLayouts": [{"config": "lvm", "mountpoint": "/", "disks": ["x"]}]}
        assert get_root_disk_gib(d) is None
        assert get_root_disk_gib({}) is None
