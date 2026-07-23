from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np


_DAY_PER_YEAR = 365.25
_JD_OFFSET = 2450000.0
_TABLE_NAME = "earth_orbital_parallax_table.txt"


def calc_vEarth(t_jd, ra_deg, dec_deg):
    """Return projected Earth velocity as ``(v_N, v_E)`` in AU/yr.

    The ephemeris is interpolated from the packaged JPL Horizons Earth table.
    For compatibility with earlier gapmoe releases, inputs smaller than
    2450000 are interpreted as ``JD - 2450000``.
    """

    t = float(t_jd)
    if t < _JD_OFFSET:
        t += _JD_OFFSET

    table = _earth_ephemeris_table()
    t_grid = table[:, 0]
    if t < t_grid[0] or t > t_grid[-1]:
        raise ValueError(
            f"t_jd={t_jd!r} is outside the packaged Earth ephemeris range "
            f"[{t_grid[0]}, {t_grid[-1]}]."
        )

    velocity = np.array([np.interp(t, t_grid, table[:, i]) for i in range(4, 7)])
    north, east = _north_east_basis(ra_deg, dec_deg)
    # The Horizons table contains the barycentric velocity of Earth itself.
    # Project it directly: the geocentric-to-heliocentric mapping adds the
    # Earth's velocity, not the opposite Sun/reflex velocity.
    v_north = float(np.dot(velocity, north)) * _DAY_PER_YEAR
    v_east = float(np.dot(velocity, east)) * _DAY_PER_YEAR
    return v_north, v_east


@lru_cache(maxsize=1)
def _earth_ephemeris_table():
    with resources.files("gapmoe.data").joinpath(_TABLE_NAME).open(
        "r", encoding="utf-8", errors="replace"
    ) as fp:
        rows = []
        in_block = False
        for line in fp:
            text = line.strip()
            if text.startswith("$$SOE"):
                in_block = True
                continue
            if text.startswith("$$EOE"):
                break
            if not in_block or not text:
                continue
            parts = [part.strip() for part in text.split(",")]
            if len(parts) < 8:
                continue
            try:
                rows.append(
                    (
                        float(parts[0]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    )
                )
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No ephemeris rows parsed from {_TABLE_NAME}.")
    return np.asarray(rows, dtype=np.float64)


def _north_east_basis(ra_deg, dec_deg):
    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    line_of_sight = np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    equatorial_north = np.array([0.0, 0.0, 1.0])
    east = np.cross(equatorial_north, line_of_sight)
    norm = np.linalg.norm(east)
    if norm == 0:
        raise ValueError("Cannot define east/north basis at the celestial pole.")
    east = east / norm
    north = np.cross(line_of_sight, east)
    return north, east


def hms_string_to_degrees(hms_string):
    h, m, s = map(float, hms_string.split(":"))
    hours = h + m / 60 + s / 3600
    return hours * 360 / 24


def dms_string_to_degrees(dms_string):
    sign = -1 if dms_string.startswith("-") else 1
    dms_string = dms_string.lstrip("+-")
    d, m, s = map(float, dms_string.split(":"))
    return sign * (d + m / 60 + s / 3600)
