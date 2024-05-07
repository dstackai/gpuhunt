[![](https://img.shields.io/pypi/v/gpuhunt)](https://pypi.org/project/gpuhunt/)

Easy access to GPU pricing data for major cloud providers: AWS, Azure, GCP, etc.
The catalog includes details about prices, locations, CPUs, RAM, GPUs, and spots (interruptible instances).

## Usage

```python
import gpuhunt

items = gpuhunt.query(
    min_memory=16,
    min_cpu=8,
    min_gpu_count=1,
    max_price=1.0,
)

print(*items, sep="\n")
```

List of all available filters:

* `provider`: name of the provider to filter by. If not specified, all providers will be used. One or many
* `min_cpu`: minimum number of CPUs
* `max_cpu`: maximum number of CPUs
* `min_memory`: minimum amount of RAM in GB
* `max_memory`: maximum amount of RAM in GB
* `min_gpu_count`: minimum number of GPUs
* `max_gpu_count`: maximum number of GPUs
* `gpu_name`: name of the GPU to filter by. If not specified, all GPUs will be used. One or many
* `min_gpu_memory`: minimum amount of GPU VRAM in GB for each GPU
* `max_gpu_memory`: maximum amount of GPU VRAM in GB for each GPU
* `min_total_gpu_memory`: minimum amount of GPU VRAM in GB for all GPUs combined
* `max_total_gpu_memory`: maximum amount of GPU VRAM in GB for all GPUs combined
* `min_disk_size`: minimum disk size in GB (not fully supported)
* `max_disk_size`: maximum disk size in GB (not fully supported)
* `min_price`: minimum price per hour in USD
* `max_price`: maximum price per hour in USD
* `min_compute_capability`: minimum compute capability of the GPU
* `max_compute_capability`: maximum compute capability of the GPU
* `spot`: if `False`, only ondemand offers will be returned. If `True`, only spot offers will be returned

## Advanced usage

```python
from gpuhunt import Catalog

catalog = Catalog()
catalog.load(version="20240508")
items = catalog.query()

print(*items, sep="\n")
```

## Supported providers

* AWS
* Azure
* Cudo Compute
* DataCrunch
* GCP
* LambdaLabs
* OCI
* RunPod
* TensorDock
* Vast AI

## See also

* [dstack](https://github.com/dstackai/dstack)
