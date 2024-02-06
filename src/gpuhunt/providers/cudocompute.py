import asyncio
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Optional, List
from cudo_compute import cudo_api
from gpuhunt import QueryFilter, RawCatalogItem
from gpuhunt._internal.constraints import KNOWN_GPUS
from gpuhunt.providers import AbstractProvider
import logging

CpuMemoryGpu = namedtuple("CpuMemoryGpu", ["cpu", "memory", "gpu"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CudoComputeProvider(AbstractProvider):
    NAME = "cudocompute"

    def get(
            self, query_filter: Optional[QueryFilter] = None, balance_resources: bool = True
    ) -> List[RawCatalogItem]:
        offers = asyncio.run(self.fetch_all_vm_types())
        return sorted(offers, key=lambda i: i.price)

    async def fetch_all_vm_types(self):
        executor = ThreadPoolExecutor(max_workers=10)
        tasks = []
        for cmg in GPU_MACHINES:
            task = asyncio.create_task(self.fetch_vm_type(cmg.cpu, cmg.memory, cmg.gpu, executor))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        executor.shutdown(wait=True)
        return list(chain.from_iterable(results))

    def get_raw_catalog_list(self, vm_machine_type_list, vcpu, memory, gpu):
        raw_list = []
        for vm in vm_machine_type_list.host_configs:
            raw = RawCatalogItem(
                instance_name=vm.machine_type,
                location=vm.data_center_id,
                spot=False,
                price=round((float(vm.total_price_hr.value) + float(vm.storage_gib_price_hr.value)), 5),
                cpu=vcpu,
                memory=memory,
                gpu_count=gpu,
                gpu_name=gpu_name(vm.gpu_model),
                gpu_memory=get_memory(gpu_name(vm.gpu_model)),
                disk_size=None
            )
            raw_list.append(raw)
        return raw_list

    async def fetch_vm_type(self, vcpu, memory_gib, gpu, executor):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: cudo_api.virtual_machines().list_vm_machine_types(vcpu=vcpu, memory_gib=memory_gib, gpu=gpu)
        )
        return self.get_raw_catalog_list(result, vcpu, memory_gib, gpu)


def gpu_name(name: str) -> Optional[str]:
    if not name:
        return None

    result = GPU_MAP.get(name)

    if result is None:
        logging.warning("There is no '%s' in GPU_MAP", name)

    return result


def get_memory(gpu_name: str) -> Optional[int]:
    for gpu in KNOWN_GPUS:
        if gpu.name.lower() == gpu_name.lower():
            return gpu.memory
    return None


GPU_MAP = {
    "RTX A4000": "A4000",
    "RTX A4500": "A4500",
    "RTX A5000": "A5000",
    "RTX A6000": "A6000",
    "NVIDIA A40": "A40",
    "NVIDIA V100": "V100",
}

GPU_MACHINES = [
    CpuMemoryGpu(1, 1, 1),
    CpuMemoryGpu(1, 2, 1),
    CpuMemoryGpu(1, 3, 1),
    CpuMemoryGpu(1, 4, 1),
    CpuMemoryGpu(2, 2, 1),
    CpuMemoryGpu(2, 2, 2),
    CpuMemoryGpu(2, 3, 1),
    CpuMemoryGpu(2, 3, 2),
    CpuMemoryGpu(2, 4, 1),
    CpuMemoryGpu(2, 4, 2),
    CpuMemoryGpu(2, 6, 1),
    CpuMemoryGpu(2, 6, 2),
    CpuMemoryGpu(2, 8, 1),
    CpuMemoryGpu(2, 8, 2),
    CpuMemoryGpu(3, 3, 1),
    CpuMemoryGpu(3, 3, 2),
    CpuMemoryGpu(3, 3, 3),
    CpuMemoryGpu(3, 4, 1),
    CpuMemoryGpu(3, 4, 2),
    CpuMemoryGpu(3, 4, 3),
    CpuMemoryGpu(3, 6, 1),
    CpuMemoryGpu(3, 6, 2),
    CpuMemoryGpu(3, 6, 3),
    CpuMemoryGpu(3, 8, 1),
    CpuMemoryGpu(3, 8, 2),
    CpuMemoryGpu(3, 8, 3),
    CpuMemoryGpu(3, 12, 1),
    CpuMemoryGpu(3, 12, 2),
    CpuMemoryGpu(3, 12, 3),
    CpuMemoryGpu(4, 4, 1),
    CpuMemoryGpu(4, 4, 2),
    CpuMemoryGpu(4, 4, 3),
    CpuMemoryGpu(4, 4, 4),
    CpuMemoryGpu(4, 6, 1),
    CpuMemoryGpu(4, 6, 2),
    CpuMemoryGpu(4, 6, 3),
    CpuMemoryGpu(4, 6, 4),
    CpuMemoryGpu(4, 8, 1),
    CpuMemoryGpu(4, 8, 2),
    CpuMemoryGpu(4, 8, 3),
    CpuMemoryGpu(4, 8, 4),
    CpuMemoryGpu(4, 12, 1),
    CpuMemoryGpu(4, 12, 2),
    CpuMemoryGpu(4, 12, 3),
    CpuMemoryGpu(4, 12, 4),
    CpuMemoryGpu(4, 16, 1),
    CpuMemoryGpu(4, 16, 2),
    CpuMemoryGpu(4, 16, 3),
    CpuMemoryGpu(4, 16, 4),
    CpuMemoryGpu(6, 6, 1),
    CpuMemoryGpu(6, 6, 2),
    CpuMemoryGpu(6, 6, 3),
    CpuMemoryGpu(6, 6, 4),
    CpuMemoryGpu(6, 6, 5),
    CpuMemoryGpu(6, 6, 6),
    CpuMemoryGpu(6, 8, 1),
    CpuMemoryGpu(6, 8, 2),
    CpuMemoryGpu(6, 8, 3),
    CpuMemoryGpu(6, 8, 4),
    CpuMemoryGpu(6, 8, 5),
    CpuMemoryGpu(6, 8, 6),
    CpuMemoryGpu(6, 12, 1),
    CpuMemoryGpu(6, 12, 2),
    CpuMemoryGpu(6, 12, 3),
    CpuMemoryGpu(6, 12, 4),
    CpuMemoryGpu(6, 12, 5),
    CpuMemoryGpu(6, 12, 6),
    CpuMemoryGpu(6, 16, 1),
    CpuMemoryGpu(6, 16, 2),
    CpuMemoryGpu(6, 16, 3),
    CpuMemoryGpu(6, 16, 4),
    CpuMemoryGpu(6, 16, 5),
    CpuMemoryGpu(6, 16, 6),
    CpuMemoryGpu(6, 24, 1),
    CpuMemoryGpu(6, 24, 2),
    CpuMemoryGpu(6, 24, 3),
    CpuMemoryGpu(6, 24, 4),
    CpuMemoryGpu(6, 24, 5),
    CpuMemoryGpu(6, 24, 6),
    CpuMemoryGpu(8, 8, 1),
    CpuMemoryGpu(8, 8, 2),
    CpuMemoryGpu(8, 8, 3),
    CpuMemoryGpu(8, 8, 4),
    CpuMemoryGpu(8, 8, 5),
    CpuMemoryGpu(8, 8, 6),
    CpuMemoryGpu(8, 8, 7),
    CpuMemoryGpu(8, 8, 8),
    CpuMemoryGpu(8, 12, 1),
    CpuMemoryGpu(8, 12, 2),
    CpuMemoryGpu(8, 12, 3),
    CpuMemoryGpu(8, 12, 4),
    CpuMemoryGpu(8, 12, 5),
    CpuMemoryGpu(8, 12, 6),
    CpuMemoryGpu(8, 12, 7),
    CpuMemoryGpu(8, 12, 8),
    CpuMemoryGpu(8, 16, 1),
    CpuMemoryGpu(8, 16, 2),
    CpuMemoryGpu(8, 16, 3),
    CpuMemoryGpu(8, 16, 4),
    CpuMemoryGpu(8, 16, 5),
    CpuMemoryGpu(8, 16, 6),
    CpuMemoryGpu(8, 16, 7),
    CpuMemoryGpu(8, 16, 8),
    CpuMemoryGpu(8, 24, 1),
    CpuMemoryGpu(8, 24, 2),
    CpuMemoryGpu(8, 24, 3),
    CpuMemoryGpu(8, 24, 4),
    CpuMemoryGpu(8, 24, 5),
    CpuMemoryGpu(8, 24, 6),
    CpuMemoryGpu(8, 24, 7),
    CpuMemoryGpu(8, 24, 8),
    CpuMemoryGpu(8, 32, 1),
    CpuMemoryGpu(8, 32, 2),
    CpuMemoryGpu(8, 32, 3),
    CpuMemoryGpu(8, 32, 4),
    CpuMemoryGpu(8, 32, 5),
    CpuMemoryGpu(8, 32, 6),
    CpuMemoryGpu(8, 32, 7),
    CpuMemoryGpu(8, 32, 8),
    CpuMemoryGpu(12, 12, 1),
    CpuMemoryGpu(12, 12, 2),
    CpuMemoryGpu(12, 12, 3),
    CpuMemoryGpu(12, 12, 4),
    CpuMemoryGpu(12, 12, 5),
    CpuMemoryGpu(12, 12, 6),
    CpuMemoryGpu(12, 12, 7),
    CpuMemoryGpu(12, 12, 8),
    CpuMemoryGpu(12, 16, 1),
    CpuMemoryGpu(12, 16, 2),
    CpuMemoryGpu(12, 16, 3),
    CpuMemoryGpu(12, 16, 4),
    CpuMemoryGpu(12, 16, 5),
    CpuMemoryGpu(12, 16, 6),
    CpuMemoryGpu(12, 16, 7),
    CpuMemoryGpu(12, 16, 8),
    CpuMemoryGpu(12, 24, 1),
    CpuMemoryGpu(12, 24, 2),
    CpuMemoryGpu(12, 24, 3),
    CpuMemoryGpu(12, 24, 4),
    CpuMemoryGpu(12, 24, 5),
    CpuMemoryGpu(12, 24, 6),
    CpuMemoryGpu(12, 24, 7),
    CpuMemoryGpu(12, 24, 8),
    CpuMemoryGpu(12, 32, 1),
    CpuMemoryGpu(12, 32, 2),
    CpuMemoryGpu(12, 32, 3),
    CpuMemoryGpu(12, 32, 4),
    CpuMemoryGpu(12, 32, 5),
    CpuMemoryGpu(12, 32, 6),
    CpuMemoryGpu(12, 32, 7),
    CpuMemoryGpu(12, 32, 8),
    CpuMemoryGpu(12, 48, 1),
    CpuMemoryGpu(12, 48, 2),
    CpuMemoryGpu(12, 48, 3),
    CpuMemoryGpu(12, 48, 4),
    CpuMemoryGpu(12, 48, 5),
    CpuMemoryGpu(12, 48, 6),
    CpuMemoryGpu(12, 48, 7),
    CpuMemoryGpu(12, 48, 8),
    CpuMemoryGpu(16, 16, 1),
    CpuMemoryGpu(16, 16, 2),
    CpuMemoryGpu(16, 16, 3),
    CpuMemoryGpu(16, 16, 4),
    CpuMemoryGpu(16, 16, 5),
    CpuMemoryGpu(16, 16, 6),
    CpuMemoryGpu(16, 16, 7),
    CpuMemoryGpu(16, 16, 8),
    CpuMemoryGpu(16, 24, 1),
    CpuMemoryGpu(16, 24, 2),
    CpuMemoryGpu(16, 24, 3),
    CpuMemoryGpu(16, 24, 4),
    CpuMemoryGpu(16, 24, 5),
    CpuMemoryGpu(16, 24, 6),
    CpuMemoryGpu(16, 24, 7),
    CpuMemoryGpu(16, 24, 8),
    CpuMemoryGpu(16, 32, 1),
    CpuMemoryGpu(16, 32, 2),
    CpuMemoryGpu(16, 32, 3),
    CpuMemoryGpu(16, 32, 4),
    CpuMemoryGpu(16, 32, 5),
    CpuMemoryGpu(16, 32, 6),
    CpuMemoryGpu(16, 32, 7),
    CpuMemoryGpu(16, 32, 8),
    CpuMemoryGpu(16, 48, 1),
    CpuMemoryGpu(16, 48, 2),
    CpuMemoryGpu(16, 48, 3),
    CpuMemoryGpu(16, 48, 4),
    CpuMemoryGpu(16, 48, 5),
    CpuMemoryGpu(16, 48, 6),
    CpuMemoryGpu(16, 48, 7),
    CpuMemoryGpu(16, 48, 8),
    CpuMemoryGpu(16, 64, 1),
    CpuMemoryGpu(16, 64, 2),
    CpuMemoryGpu(16, 64, 3),
    CpuMemoryGpu(16, 64, 4),
    CpuMemoryGpu(16, 64, 5),
    CpuMemoryGpu(16, 64, 6),
    CpuMemoryGpu(16, 64, 7),
    CpuMemoryGpu(16, 64, 8),
    CpuMemoryGpu(24, 24, 1),
    CpuMemoryGpu(24, 24, 2),
    CpuMemoryGpu(24, 24, 3),
    CpuMemoryGpu(24, 24, 4),
    CpuMemoryGpu(24, 24, 5),
    CpuMemoryGpu(24, 24, 6),
    CpuMemoryGpu(24, 24, 7),
    CpuMemoryGpu(24, 24, 8),
    CpuMemoryGpu(24, 32, 1),
    CpuMemoryGpu(24, 32, 2),
    CpuMemoryGpu(24, 32, 3),
    CpuMemoryGpu(24, 32, 4),
    CpuMemoryGpu(24, 32, 5),
    CpuMemoryGpu(24, 32, 6),
    CpuMemoryGpu(24, 32, 7),
    CpuMemoryGpu(24, 32, 8),
    CpuMemoryGpu(24, 48, 1),
    CpuMemoryGpu(24, 48, 2),
    CpuMemoryGpu(24, 48, 3),
    CpuMemoryGpu(24, 48, 4),
    CpuMemoryGpu(24, 48, 5),
    CpuMemoryGpu(24, 48, 6),
    CpuMemoryGpu(24, 48, 7),
    CpuMemoryGpu(24, 48, 8),
    CpuMemoryGpu(24, 64, 1),
    CpuMemoryGpu(24, 64, 2),
    CpuMemoryGpu(24, 64, 3),
    CpuMemoryGpu(24, 64, 4),
    CpuMemoryGpu(24, 64, 5),
    CpuMemoryGpu(24, 64, 6),
    CpuMemoryGpu(24, 64, 7),
    CpuMemoryGpu(24, 64, 8),
    CpuMemoryGpu(24, 96, 1),
    CpuMemoryGpu(24, 96, 2),
    CpuMemoryGpu(24, 96, 3),
    CpuMemoryGpu(24, 96, 4),
    CpuMemoryGpu(24, 96, 5),
    CpuMemoryGpu(24, 96, 6),
    CpuMemoryGpu(24, 96, 7),
    CpuMemoryGpu(24, 96, 8),
    CpuMemoryGpu(32, 32, 1),
    CpuMemoryGpu(32, 32, 2),
    CpuMemoryGpu(32, 32, 3),
    CpuMemoryGpu(32, 32, 4),
    CpuMemoryGpu(32, 32, 5),
    CpuMemoryGpu(32, 32, 6),
    CpuMemoryGpu(32, 32, 7),
    CpuMemoryGpu(32, 32, 8),
    CpuMemoryGpu(32, 48, 1),
    CpuMemoryGpu(32, 48, 2),
    CpuMemoryGpu(32, 48, 3),
    CpuMemoryGpu(32, 48, 4),
    CpuMemoryGpu(32, 48, 5),
    CpuMemoryGpu(32, 48, 6),
    CpuMemoryGpu(32, 48, 7),
    CpuMemoryGpu(32, 48, 8),
    CpuMemoryGpu(32, 64, 1),
    CpuMemoryGpu(32, 64, 2),
    CpuMemoryGpu(32, 64, 3),
    CpuMemoryGpu(32, 64, 4),
    CpuMemoryGpu(32, 64, 5),
    CpuMemoryGpu(32, 64, 6),
    CpuMemoryGpu(32, 64, 7),
    CpuMemoryGpu(32, 64, 8),
    CpuMemoryGpu(32, 96, 1),
    CpuMemoryGpu(32, 96, 2),
    CpuMemoryGpu(32, 96, 3),
    CpuMemoryGpu(32, 96, 4),
    CpuMemoryGpu(32, 96, 5),
    CpuMemoryGpu(32, 96, 6),
    CpuMemoryGpu(32, 96, 7),
    CpuMemoryGpu(32, 96, 8),
    CpuMemoryGpu(32, 128, 2),
    CpuMemoryGpu(32, 128, 3),
    CpuMemoryGpu(32, 128, 4),
    CpuMemoryGpu(32, 128, 5),
    CpuMemoryGpu(32, 128, 6),
    CpuMemoryGpu(32, 128, 7),
    CpuMemoryGpu(32, 128, 8),
    CpuMemoryGpu(48, 48, 2),
    CpuMemoryGpu(48, 48, 3),
    CpuMemoryGpu(48, 48, 4),
    CpuMemoryGpu(48, 48, 5),
    CpuMemoryGpu(48, 48, 6),
    CpuMemoryGpu(48, 48, 7),
    CpuMemoryGpu(48, 48, 8),
    CpuMemoryGpu(48, 64, 2),
    CpuMemoryGpu(48, 64, 3),
    CpuMemoryGpu(48, 64, 4),
    CpuMemoryGpu(48, 64, 5),
    CpuMemoryGpu(48, 64, 6),
    CpuMemoryGpu(48, 64, 7),
    CpuMemoryGpu(48, 64, 8),
    CpuMemoryGpu(48, 96, 2),
    CpuMemoryGpu(48, 96, 3),
    CpuMemoryGpu(48, 96, 4),
    CpuMemoryGpu(48, 96, 5),
    CpuMemoryGpu(48, 96, 6),
    CpuMemoryGpu(48, 96, 7),
    CpuMemoryGpu(48, 96, 8),
    CpuMemoryGpu(48, 128, 2),
    CpuMemoryGpu(48, 128, 3),
    CpuMemoryGpu(48, 128, 4),
    CpuMemoryGpu(48, 128, 5),
    CpuMemoryGpu(48, 128, 6),
    CpuMemoryGpu(48, 128, 7),
    CpuMemoryGpu(48, 128, 8),
    CpuMemoryGpu(48, 192, 2),
    CpuMemoryGpu(48, 192, 3),
    CpuMemoryGpu(48, 192, 4),
    CpuMemoryGpu(48, 192, 5),
    CpuMemoryGpu(48, 192, 6),
    CpuMemoryGpu(48, 192, 7),
    CpuMemoryGpu(48, 192, 8),
    CpuMemoryGpu(64, 64, 2),
    CpuMemoryGpu(64, 64, 3),
    CpuMemoryGpu(64, 64, 4),
    CpuMemoryGpu(64, 64, 5),
    CpuMemoryGpu(64, 64, 6),
    CpuMemoryGpu(64, 64, 7),
    CpuMemoryGpu(64, 64, 8),
    CpuMemoryGpu(64, 96, 2),
    CpuMemoryGpu(64, 96, 3),
    CpuMemoryGpu(64, 96, 4),
    CpuMemoryGpu(64, 96, 5),
    CpuMemoryGpu(64, 96, 6),
    CpuMemoryGpu(64, 96, 7),
    CpuMemoryGpu(64, 96, 8),
    CpuMemoryGpu(64, 128, 2),
    CpuMemoryGpu(64, 128, 3),
    CpuMemoryGpu(64, 128, 4),
    CpuMemoryGpu(64, 128, 5),
    CpuMemoryGpu(64, 128, 6),
    CpuMemoryGpu(64, 128, 7),
    CpuMemoryGpu(64, 128, 8),
    CpuMemoryGpu(64, 192, 2),
    CpuMemoryGpu(64, 192, 4),
    CpuMemoryGpu(64, 192, 5),
    CpuMemoryGpu(64, 192, 6),
    CpuMemoryGpu(64, 192, 7),
    CpuMemoryGpu(64, 192, 8),
    CpuMemoryGpu(64, 256, 4),
    CpuMemoryGpu(64, 256, 5),
    CpuMemoryGpu(64, 256, 6),
    CpuMemoryGpu(64, 256, 7),
    CpuMemoryGpu(64, 256, 8),
    CpuMemoryGpu(96, 96, 4),
    CpuMemoryGpu(96, 96, 6),
    CpuMemoryGpu(96, 96, 7),
    CpuMemoryGpu(96, 96, 8),
    CpuMemoryGpu(96, 128, 4),
    CpuMemoryGpu(96, 128, 6),
    CpuMemoryGpu(96, 128, 7),
    CpuMemoryGpu(96, 128, 8),
    CpuMemoryGpu(96, 192, 6),
    CpuMemoryGpu(96, 192, 7),
    CpuMemoryGpu(96, 192, 8),
    CpuMemoryGpu(96, 256, 6),
    CpuMemoryGpu(96, 256, 7),
    CpuMemoryGpu(96, 256, 8),
    CpuMemoryGpu(96, 384, 6),
    CpuMemoryGpu(96, 384, 7),
    CpuMemoryGpu(96, 384, 8),
    CpuMemoryGpu(128, 128, 8),
    CpuMemoryGpu(128, 192, 8),
    CpuMemoryGpu(128, 256, 8),
    CpuMemoryGpu(128, 384, 8)
]
