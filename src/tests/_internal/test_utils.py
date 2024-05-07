import pytest

from gpuhunt._internal.utils import to_camel_case


@pytest.mark.parametrize(
    ["before", "after"],
    [
        ["spam_ham_eggs", "spamHamEggs"],
        ["spam__ham__eggs", "spamHamEggs"],
        ["__spam_ham_eggs__", "spamHamEggs"],
        ["spamHam_eggs", "spamHamEggs"],
        ["spamHamEggs", "spamHamEggs"],
        ["SpamHam_eggs", "SpamHamEggs"],
        ["spam", "spam"],
        ["", ""],
    ],
)
def test_to_camel_case(before, after):
    assert to_camel_case(before) == after
