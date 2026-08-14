class GPUHuntError(Exception):
    pass


class ProviderError(GPUHuntError):
    pass


class MissingCredsError(ProviderError):
    """
    Raised by `OnlineProvider.from_env` when a required environment variable is not set.
    Online providers are skipped rather than failing the whole catalog when this is raised.
    """
