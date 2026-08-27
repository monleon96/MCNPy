"""The angle frame must come from X4Pro's ifCM flag, not from the reaction code.

Regression guard for the 2026-08-26 fix. X4Pro normally converts a CM-quoted
angle column to the lab frame and leaves ``ifCM`` False, so reading the frame
off the reaction code happened to work for almost every dataset. It does not
when the *cross section* is CM as well (EXFOR header ``DATA-CM`` against
``COS-CM``): X4Pro then leaves both in the centre of mass and says so only
through ``ifCM``. Becker 1966 (11511009) is the one such dataset in the Fe-56
angular corpus, and it was being labelled lab and sent through a second
lab->CM transform downstream.

The fixtures below are the real shapes of the two cases, cut down to three
points. They are written to a temporary sqlite file because ``X4ProDatabase``
takes a path, and they carry only the two columns the loader reads.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from kika.exfor.database import X4ProDatabase, _parse_c5data_json


def _dataset_json(*, header: str, if_cm: bool, angles, values) -> dict:
    """Minimal jx5z payload with one energy and one angle variable."""
    return {
        "c5data": {
            "y": {"y": values, "dy": [0.01] * len(values), "units": "B/SR"},
            "x1": {"fam": "EN", "x1": [3.2e6] * len(values), "units": "EV"},
            "x2": {
                "fam": "ANG",
                "header": header,
                "units": "ADEG",
                "ifCM": if_cm,
                "x2": angles,
            },
        },
        "x4data": [],
    }


#: (DatasetID, reacode, ifCM, angles) — Becker's CM pair and an ordinary lab set.
FIXTURES = [
    ("11511009", "26-FE-0(N,EL),,DA", True, [20.397, 30.503, 40.598]),
    ("10886002", "26-FE-0(N,EL)26-FE-0,,DA", False, [20.3, 30.2, 46.1]),
]


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("x4") / "frames.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE x4pro_ds (DatasetID TEXT, year1 INT, author1 TEXT, "
        "Targ1 TEXT, Proj TEXT, MF INT, MT INT, ndat INT, quant1 TEXT, reacode TEXT)"
    )
    conn.execute("CREATE TABLE x4pro_x5z (DatasetID TEXT, jx5z TEXT)")
    for ds, reacode, if_cm, angles in FIXTURES:
        conn.execute(
            "INSERT INTO x4pro_ds VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ds, 1966, "Becker", "Fe-0", "N", 154, 2, len(angles), "DA", reacode),
        )
        payload = _dataset_json(
            header="ANG", if_cm=if_cm, angles=angles, values=[1.0, 0.7, 0.4],
        )
        conn.execute(
            "INSERT INTO x4pro_x5z VALUES (?,?)", (ds, json.dumps(payload))
        )
    conn.commit()
    conn.close()
    return str(path)


def test_ifcm_true_is_reported_as_cm(db_path):
    """The dataset whose c5 angle carries ifCM=True must load as CM.

    Its reaction code (``26-FE-0(N,EL),,DA``) contains no frame marker at all,
    which is exactly why the pre-fix rule got it wrong.
    """
    ds = X4ProDatabase(db_path).parse_dataset("11511009")
    assert ds is not None
    assert ds.angle_frame == "CM"
    # The angles must arrive untouched: nothing may "helpfully" convert them.
    assert np.allclose(ds.angles_deg, [20.397, 30.503, 40.598])


def test_ifcm_false_is_still_lab(db_path):
    """The ordinary case must not move — this is the 99% path."""
    ds = X4ProDatabase(db_path).parse_dataset("10886002")
    assert ds is not None
    assert ds.angle_frame == "LAB"


def test_parser_surfaces_ifcm():
    """``is_cm`` defaults to False and mirrors the flag when present."""
    with_flag = _parse_c5data_json(
        _dataset_json(header="ANG", if_cm=True, angles=[30.0], values=[1.0])
    )
    assert with_flag["is_cm"] is True

    without_flag = _parse_c5data_json(
        _dataset_json(header="ANG", if_cm=False, angles=[30.0], values=[1.0])
    )
    assert without_flag["is_cm"] is False

    # A payload with no x2 at all must not raise and must not claim CM.
    assert _parse_c5data_json({"c5data": {}})["is_cm"] is False
