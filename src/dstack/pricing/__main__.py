import argparse
import logging
import os
import sys

import dstack.pricing.storage as storage
from dstack.pricing.providers.aws import AWSProvider
from dstack.pricing.providers.azure import AzureProvider
from dstack.pricing.providers.gcp import GCPProvider
from dstack.pricing.providers.lambdalabs import LambdaLabsProvider
from dstack.pricing.providers.tensordock import TensorDockProvider
from dstack.pricing.providers.vastai import VastAIProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=["aws", "azure", "gcp", "lambdalabs", "tensordock", "vastai"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-filter", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")

    if args.provider == "aws":
        provider = AWSProvider()
    elif args.provider == "azure":
        provider = AzureProvider(os.getenv("AZURE_SUBSCRIPTION_ID"))
    elif args.provider == "gcp":
        provider = GCPProvider(os.getenv("GCP_PROJECT_ID"))
    elif args.provider == "lambdalabs":
        provider = LambdaLabsProvider(os.getenv("LAMBDALABS_TOKEN"))
    elif args.provider == "tensordock":
        provider = TensorDockProvider()
    elif args.provider == "vastai":
        provider = VastAIProvider()
    else:
        exit(f"Unknown provider {args.provider}")

    offers = provider.get()
    if not args.no_filter:
        offers = provider.filter(offers)
    storage.dump(offers, args.output)


if __name__ == "__main__":
    main()
