"""utility functions"""

from fmtda.parse_dict import get_abbrev_map

abbrev2desc, _ = get_abbrev_map()

ALL_FEATURES = ["gp"]
for abbrev, desc in abbrev2desc.items():
    for word in desc.lower().split(" "):
        if (
            word == "left"
            or word == "right"
            or word == "arm"
            or word == "leg"
            or word == "gp"
            or word == "upper"
            or word == "lower"
        ):
            ALL_FEATURES.append(abbrev)
            break
