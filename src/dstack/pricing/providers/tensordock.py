import math

import requests

from dstack.pricing.models import InstanceOffer
from dstack.pricing.providers import AbstractProvider


# https://documenter.getpostman.com/view/10732984/UVC3j7Kz#f1eeb07f-294d-4a7a-b7d9-b3ede1605c01
core_cloud_instances_url = "https://console.tensordock.com/api/metadata/instances"
# https://documenter.getpostman.com/view/20973002/2s8YzMYRDc#2b4a3db0-c162-438c-aae4-6a88afc96fdb
marketplace_hostnodes_url = "https://marketplace.tensordock.com/api/v0/client/deploy/hostnodes"


class TensorDockProvider(AbstractProvider):
    def get_core_cloud(self) -> list[InstanceOffer]:
        data = requests.get(core_cloud_instances_url).json()
        offers = []
        for name, resources in data["cpu"].items():
            for location in resources["locations"]:
                for multiples in range(1, resources["restrictions"]["maxMultiples"] + 1):
                    offer = InstanceOffer(
                        instance_name=f"{name.lower()}_{multiples}",
                        location=location,
                        price=round(multiples * resources["cost"]["costHr"], 5),
                        cpu=multiples,
                        memory=multiples * resources["specs"]["ram"],
                        gpu_count=0,
                        spot=False,
                    )
                    offers.append(offer)

        cpu_price = data["resources"]["gpu_instances"]["vcpu"]["costHr"]
        memory_price = data["resources"]["gpu_instances"]["ram"]["costHr"]
        for name, resources in data["gpu"].items():
            gpu_name = name.lower().replace("quadro_", "rtx").split("_")[0].upper()
            restrictions = resources["restrictions"]
            for location in resources["locations"]:
                for gpu_count in range(1, restrictions["maxGPUsPerInstance"] + 1):
                    memory = min(2 * gpu_count * resources["specs"]["vram"], restrictions["maxRAMPerInstance"])
                    min_cpu = math.ceil(memory / restrictions["maxRAMPervCPU"])
                    cpu = min(max(min_cpu, gpu_count * min(12, restrictions["maxvCPUsPerGPU"])), restrictions["maxvCPUsPerInstance"])
                    offer = InstanceOffer(
                        instance_name=f"{name.lower()}_{gpu_count}_{cpu}_{memory}",
                        location=location,
                        price=round(gpu_count * resources["cost"]["costHr"] + cpu * cpu_price + memory * memory_price, 5),
                        cpu=cpu,
                        memory=memory,
                        gpu_count=gpu_count,
                        gpu_name=gpu_name,
                        gpu_memory=resources["specs"]["vram"],
                        spot=False,
                    )
                    offers.append(offer)
        return offers

    def get(self) -> list[InstanceOffer]:
        return self.get_core_cloud()
