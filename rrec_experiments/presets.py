"""Explicit dataset profiles; sampling interventions share each profile unchanged."""

PRESETS = {
    "yelp2018": {"ssl_weight": .2, "temperature": .2, "drop_ratio": .1, "decay": 1e-4},
    "amazon-book": {"ssl_weight": .5, "temperature": .2, "drop_ratio": .1, "decay": 1e-4},
    "gowalla": {"ssl_weight": .3, "temperature": .2, "drop_ratio": .2, "decay": 1e-4},
    "iFashion": {"ssl_weight": .02, "temperature": .5, "drop_ratio": .4, "decay": 1e-3},
}


def preset_options(dataset):
    return [value for name, setting in PRESETS[dataset].items()
            for value in ("--" + name.replace("_", "-"), str(setting))]
