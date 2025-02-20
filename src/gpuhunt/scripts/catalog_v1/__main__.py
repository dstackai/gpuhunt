import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Optional

from gpuhunt._internal import storage
from gpuhunt._internal.utils import configure_logging


def main(args: Optional[Sequence[str]] = None):
    configure_logging()
    parser = argparse.ArgumentParser(
        description=dedent(
            """
            Convert a v2 catalog to a v1 catalog. Legacy v1 catalogs are used by older
            gpuhunt versions that do not respect the `flags` field. Any catalog items
            with flags are filtered out when converting to v1.
            """
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="The v2 catalog file to read")
    parser.add_argument("--output", type=Path, required=True, help="The v1 catalog file to write")
    args = parser.parse_args(args)
    storage.convert_catalog_v2_to_v1(path_v2=args.input, path_v1=args.output)
    logging.info("Converted %s -> %s", args.input, args.output)


if __name__ == "__main__":
    main()
