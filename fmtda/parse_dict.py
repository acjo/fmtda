"""Parsing module."""

from pathlib import Path

import pandas as pd


def get_abbrev_map() -> tuple[dict, dict]:
    """Create abbreviation to description and desciription to abbreviation dict.

    Parameters
    ----------
    None

    Returns
    -------
    abbrev2desc : dict
        Abbreviation to description dictionary
    desc2abbrev : dict
        Description to abbreviation dictionary
    """
    path = Path(__file__)
    parent = path.parent
    data_path = parent / "Clinical_fm_66_.xlsx"

    data = pd.read_excel(data_path, sheet_name="Dictionary")

    abbrev2desc = data.loc[:, ["Abbreviation", "Description"]].to_numpy()
    abbrev2desc = dict(abbrev2desc)

    desc2abbrev = data.loc[:, ["Description", "Abbreviation"]].to_numpy()
    desc2abbrev = dict(desc2abbrev)

    return abbrev2desc, desc2abbrev
