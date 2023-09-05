Easy access to GPU pricing data for major cloud providers: AWS, Azure, GCP, and LambdaLabs.
The catalog includes details about prices, locations, CPUs, RAM, GPUs, and spots (interruptable instances).

## Usage

```python
import dstack.pricing

gpus = dstack.pricing.query()
dstack.pricing.print_table(gpus)
```

## Advanced usage

```python
import dstack.pricing

catalog = dstack.pricing.Catalog()
catalog.load()
gpus = catalog.query()
dstack.pricing.print_table(gpus)
```

## See also

* [dstack](https://github.com/dstackai/dstack)
