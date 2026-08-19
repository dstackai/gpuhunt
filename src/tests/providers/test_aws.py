from gpuhunt.providers.aws import _prefilter_rows

HEADER = (
    '"TermType","Tenancy","Operating System","CapacityStatus",'
    '"Unit","Currency","Pre Installed S/W","MarketOption"\n'
)
ON_DEMAND_ROW = '"OnDemand","Shared","Linux","Used","Hrs","USD",,"OnDemand"\n'
RESERVED_ROW = '"Reserved","Shared","Linux","Used","Hrs","USD",,\n'
WINDOWS_ROW = '"OnDemand","Shared","Windows","Used","Hrs","USD",,"OnDemand"\n'


class TestPrefilterRows:
    def test_keeps_header_and_rows_matching_pricing_filters(self):
        rows = [HEADER, ON_DEMAND_ROW, RESERVED_ROW, WINDOWS_ROW]
        assert list(_prefilter_rows(rows)) == [HEADER, ON_DEMAND_ROW]

    def test_handles_empty_input(self):
        assert list(_prefilter_rows([])) == []
