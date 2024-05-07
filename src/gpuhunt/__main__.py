import argparse
import logging
import os
import sys

import gpuhunt._internal.storage as storage


def main():
    parser = argparse.ArgumentParser(prog="python3 -m gpuhunt")
    parser.add_argument(
        "provider",
        choices=[
            "aws",
            "azure",
            "cudo",
            "datacrunch",
            "gcp",
            "lambdalabs",
            "oci",
            "runpod",
            "tensordock",
            "vastai",
        ],
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
    elif args.provider == "cudo":
        from gpuhunt.providers.cudo import CudoProvider

        provider = CudoProvider()
    elif args.provider == "datacrunch":
        from gpuhunt.providers.datacrunch import DataCrunchProvider

        provider = DataCrunchProvider(
            os.getenv("DATACRUNCH_CLIENT_ID"), os.getenv("DATACRUNCH_CLIENT_SECRET")
        )
    elif args.provider == "gcp":
        from gpuhunt.providers.gcp import GCPProvider

        provider = GCPProvider(os.getenv("GCP_PROJECT_ID"))
    elif args.provider == "lambdalabs":
        from gpuhunt.providers.lambdalabs import LambdaLabsProvider

        provider = LambdaLabsProvider(os.getenv("LAMBDALABS_TOKEN"))
    elif args.provider == "oci":
        from gpuhunt.providers.oci import OCICredentials, OCIProvider

        provider = OCIProvider(
            OCICredentials(
                user=os.getenv("OCI_CLI_USER"),
                key_content=os.getenv("OCI_CLI_KEY_CONTENT"),
                fingerprint=os.getenv("OCI_CLI_FINGERPRINT"),
                tenancy=os.getenv("OCI_CLI_TENANCY"),
                region=os.getenv("OCI_CLI_REGION"),
            )
        )
    elif args.provider == "runpod":
        from gpuhunt.providers.runpod import RunpodProvider

        provider = RunpodProvider()
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
