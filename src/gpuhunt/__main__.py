import argparse
import logging
import os
import sys

import gpuhunt._internal.storage as storage


def main():
    parser = argparse.ArgumentParser(prog="python3 -m gpuhunt")
    parser.add_argument(
        "provider",
        choices=["aws", "azure", "gcp", "lambdalabs", "tensordock", "vastai"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-filter", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.provider == "aws":
        from gpuhunt.providers.aws import AWSProvider

        provider = AWSProvider(os.getenv("AWS_CACHE_PATH"))
    elif args.provider == "azure":
        from gpuhunt.providers.azure import AzureProvider

        provider = AzureProvider(os.getenv("AZURE_SUBSCRIPTION_ID"))
    elif args.provider == "gcp":
        from gpuhunt.providers.gcp import GCPProvider

        provider = GCPProvider(os.getenv("GCP_PROJECT_ID"))
    elif args.provider == "lambdalabs":
        from gpuhunt.providers.lambdalabs import LambdaLabsProvider

        provider = LambdaLabsProvider(os.getenv("LAMBDALABS_TOKEN"))
    elif args.provider == "tensordock":
        from gpuhunt.providers.tensordock import TensorDockProvider

        provider = TensorDockProvider()
    elif args.provider == "vastai":
        from gpuhunt.providers.vastai import VastAIProvider

        provider = VastAIProvider()
    else:
        exit(f"Unknown provider {args.provider}")

    logging.info("Fetching offers for %s", args.provider)
    offers = provider.get()
    if not args.no_filter:
        offers = provider.filter(offers)
    storage.dump(offers, args.output)


if __name__ == "__main__":
    main()
