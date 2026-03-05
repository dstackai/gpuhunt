from gpuhunt._internal.models import QueryFilter
from gpuhunt.providers.vastai import VastAIProvider


def test_make_filters_defaults_to_datacenter_only():
    filters = VastAIProvider.make_filters(QueryFilter())
    assert filters["datacenter"]["eq"] is True
    assert "external" not in filters


def test_make_filters_does_not_constrain_scope_when_community_cloud_enabled():
    filters = VastAIProvider.make_filters(QueryFilter(), community_cloud=True)
    assert "datacenter" not in filters
    assert "external" not in filters
