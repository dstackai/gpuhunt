import argparse
import logging
import os

import gpuhunt._internal.storage as storage
from gpuhunt._internal.utils import configure_logging
from gpuhunt.providers.base import OfflineProvider


def main():
    parser = argparse.ArgumentParser(prog="python3 -m gpuhunt")
    parser.add_argument(
        "provider",
        choices=[
            "aws",
            "azure",
            "cloudrift",
            "crusoe",
            "verda",
            "digitalocean",
            "gcp",
            "hotaisle",
            "jarvislabs",
            "lambdalabs",
            "nebius",
            "oci",
            "runpod",
            "seeweb",
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

        provider = AzureProvider(os.environ["AZURE_SUBSCRIPTION_ID"])
    elif args.provider == "crusoe":
        from gpuhunt.providers.crusoe import CrusoeProvider

        provider = CrusoeProvider.from_env()
    elif args.provider == "cloudrift":
        from gpuhunt.providers.cloudrift import CloudRiftProvider

        provider = CloudRiftProvider()
    elif args.provider == "verda":
        from gpuhunt.providers.verda import VerdaProvider

        provider = VerdaProvider(
            client_id=os.environ["VERDA_CLIENT_ID"],
            client_secret=os.environ["VERDA_CLIENT_SECRET"],
        )
    elif args.provider == "digitalocean":
        from gpuhunt.providers.digitalocean import DigitalOceanProvider

        provider = DigitalOceanProvider.from_env()
    elif args.provider == "gcp":
        from gpuhunt.providers.gcp import GCPProvider

        provider = GCPProvider(project=os.environ["GCP_PROJECT_ID"])
    elif args.provider == "hotaisle":
        from gpuhunt.providers.hotaisle import HotAisleProvider

        provider = HotAisleProvider.from_env()
    elif args.provider == "jarvislabs":
        from gpuhunt.providers.jarvislabs import JarvisLabsProvider

        provider = JarvisLabsProvider.from_env()
    elif args.provider == "lambdalabs":
        from gpuhunt.providers.lambdalabs import LambdaLabsProvider

        provider = LambdaLabsProvider(token=os.environ["LAMBDALABS_TOKEN"])
    elif args.provider == "nebius":
        from nebius.base.service_account.pk_file import Reader as PKReader

        from gpuhunt.providers.nebius import NebiusProvider

        provider = NebiusProvider(
            credentials=(
                # temporary user token from `nebius iam get-access-token`
                os.getenv("NEBIUS_ACCESS_TOKEN")
                # or service account credentials
                or PKReader(
                    filename=os.environ["NEBIUS_PRIVATE_KEY_FILE"],
                    public_key_id=os.environ["NEBIUS_PUBLIC_KEY_ID"],
                    service_account_id=os.environ["NEBIUS_SERVICE_ACCOUNT_ID"],
                )
            )
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
    elif args.provider == "seeweb":
        from gpuhunt.providers.seeweb import SeewebProvider

        provider = SeewebProvider.from_env()
    elif args.provider == "vastai":
        from gpuhunt.providers.vastai import VastAIProvider

        provider = VastAIProvider.from_env()
    elif args.provider == "vultr":
        from gpuhunt.providers.vultr import VultrProvider

        provider = VultrProvider.from_env()
    else:
        exit(f"Unknown provider {args.provider}")

    logging.info("Fetching offers for %s", args.provider)
    offers = provider.get()
    if not args.no_filter and isinstance(provider, OfflineProvider):
        offers = provider.filter(offers)
    storage.dump(offers, args.output)


if __name__ == "__main__":
    main()
