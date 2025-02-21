from pathlib import Path
from textwrap import dedent

import pytest

from gpuhunt.scripts.catalog_v1.__main__ import main


@pytest.mark.parametrize(
    ("v2_catalog", "v1_catalog"),
    [
        pytest.param(
            dedent(
                """
                instance_name,location,price,cpu,memory,gpu_count,gpu_name,gpu_memory,spot,disk_size,gpu_vendor,flags
                i1,us,0.5,30,240,1,A10,24.0,False,,nvidia,
                i2,us,1.0,30,240,2,A10,24.0,False,,nvidia,f1
                i3,us,2.0,30,240,4,A10,24.0,False,,nvidia,f1 f2
                i4,us,4.0,30,240,8,A10,24.0,False,,nvidia,
                """
            ).lstrip(),
            dedent(
                """
                instance_name,location,price,cpu,memory,gpu_count,gpu_name,gpu_memory,spot,disk_size,gpu_vendor
                i1,us,0.5,30,240,1,A10,24.0,False,,nvidia
                i4,us,4.0,30,240,8,A10,24.0,False,,nvidia
                """
            ).lstrip(),
            id="filters-out-offers-with-flags",
        ),
        pytest.param(
            dedent(
                """
                new_column,instance_name,location,price,cpu,memory,gpu_count,gpu_name,gpu_memory,spot,disk_size,gpu_vendor,flags
                ???,i1,us,0.5,30,240,1,A10,24.0,False,,nvidia,
                ???,i2,us,1.0,30,240,2,A10,24.0,False,,nvidia,
                """
            ).lstrip(),
            dedent(
                """
                instance_name,location,price,cpu,memory,gpu_count,gpu_name,gpu_memory,spot,disk_size,gpu_vendor
                i1,us,0.5,30,240,1,A10,24.0,False,,nvidia
                i2,us,1.0,30,240,2,A10,24.0,False,,nvidia
                """
            ).lstrip(),
            id="removes-extra-columns",
        ),
    ],
)
def test_main(tmp_path: Path, v2_catalog: str, v1_catalog: str) -> None:
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    v1_file = tmp_path / "v1" / "catalog.csv"
    v2_file = tmp_path / "v2" / "catalog.csv"
    v2_file.write_text(v2_catalog)
    main(["--input", str(v2_file), "--output", str(v1_file)])
    assert v1_file.read_text() == v1_catalog
