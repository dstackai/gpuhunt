import argparse
import logging
import os

import gpuhunt._internal.storage as storage
from gpuhunt._internal.utils import configure_logging


def main():
    parser = argparse.ArgumentParser(prog="python3 -m gpuhunt")
    parser.add_argument(
        "provider",
        choices=[
            "aws",
            "azure",
            "cloudrift",
            "cudo",
            "verda",
            "digitalocean",
            "gcp",
            "hotaisle",
            "lambdalabs",
            "nebius",
            "oci",
            "runpod",
            "tensordock",
            "vastai",
            "vultr",
        ],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-filter", action="store_true")
    args = parser.parse_args()
    configure_logging()

    if args.provider == "aws":
        from gpuhunt.providers.aws import AWSProvider

        provider = AWSProvider(os.getenv("AWS_CACHE_PATH"))
    elif args.provider == "azure":
        from gpuhunt.providers.azure import AzureProvider

        provider = AzureProvider(os.getenv("AZURE_SUBSCRIPTION_ID"))
    elif args.provider == "cudo":
        from gpuhunt.providers.cudo import CudoProvider

        provider = CudoProvider()
    elif args.provider == "cloudrift":
        from gpuhunt.providers.cloudrift import CloudRiftProvider

        provider = CloudRiftProvider()
    elif args.provider == "verda":
        from gpuhunt.providers.verda import VerdaProvider

        provider = VerdaProvider(os.getenv("VERDA_CLIENT_ID"), os.getenv("VERDA_CLIENT_SECRET"))
    elif args.provider == "digitalocean":
        from gpuhunt.providers.digitalocean import DigitalOceanProvider

        provider = DigitalOceanProvider(
            api_key=os.getenv("DIGITAL_OCEAN_API_KEY"), api_url=os.getenv("DIGITAL_OCEAN_API_URL")
        )
    elif args.provider == "gcp":
        from gpuhunt.providers.gcp import GCPProvider

        provider = GCPProvider(os.getenv("GCP_PROJECT_ID"))
    elif args.provider == "hotaisle":
        from gpuhunt.providers.hotaisle import HotAisleProvider

        provider = HotAisleProvider(
            api_key=os.getenv("HOTAISLE_API_KEY"), team_handle=os.getenv("HOTAISLE_TEAM_HANDLE")
        )
    elif args.provider == "lambdalabs":
        from gpuhunt.providers.lambdalabs import LambdaLabsProvider

        provider = LambdaLabsProvider(os.getenv("LAMBDALABS_TOKEN"))
    elif args.provider == "nebius":
        from nebius.base.service_account.pk_file import Reader as PKReader

        from gpuhunt.providers.nebius import NebiusProvider

        provider = NebiusProvider(
            credentials=PKReader(
                filename=os.getenv("NEBIUS_PRIVATE_KEY_FILE"),
                public_key_id=os.getenv("NEBIUS_PUBLIC_KEY_ID"),
                service_account_id=os.getenv("NEBIUS_SERVICE_ACCOUNT_ID"),
            ),
        )
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
    elif args.provider == "vultr":
        from gpuhunt.providers.vultr import VultrProvider

        provider = VultrProvider()
    else:
        exit(f"Unknown provider {args.provider}")

    logging.info("Fetching offers for %s", args.provider)
    offers = provider.get()
    if not args.no_filter:
        offers = provider.filter(offers)
    storage.dump(offers, args.output)


if __name__ == "__main__":
    main()
