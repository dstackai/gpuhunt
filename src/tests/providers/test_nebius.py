from unittest.mock import MagicMock

from gpuhunt.providers import nebius


def test_user_agent_prefix(mocker):
    sdk_cls = mocker.patch.object(nebius, "SDK")
    mocker.patch.object(nebius, "CalculatorServiceClient")
    mocker.patch.object(nebius, "get_sample_projects", return_value={})
    mocker.patch.object(nebius, "__version__", "1.2.3")
    credentials = MagicMock()

    nebius.NebiusProvider(credentials).get()

    sdk_cls.assert_called_once_with(
        credentials=credentials,
        user_agent_prefix="gpuhunt/1.2.3",
    )
