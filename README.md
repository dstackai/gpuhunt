Easy access to GPU pricing data for major cloud providers: AWS, Azure, GCP, and LambdaLabs.
The catalog includes details about prices, locations, CPUs, RAM, GPUs, and spots (interruptable instances).

## Usage

```python
import gpuhunt

items = gpuhunt.query()

print(*items, sep="\n")
```

## Advanced usage

```python
from gpuhunt import Catalog

catalog = Catalog()
catalog.load(version="20230831")
items = catalog.query()

print(*items, sep="\n")
```

## See also

* [dstack](https://github.com/dstackai/dstack)
