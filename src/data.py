"""Dataset selection and loading (TDC ADMET benchmark group).

The §4 preregistered scaling rule: retain all four toxicity endpoints plus
ADME-category coverage with >=1 regression and >=1 classification per category
where the category supports both, for >=12 endpoints. Selection is by task type,
category coverage and size only -- never by any reliability result.

For each endpoint we load BOTH the scaffold split (primary; deployment-realistic
shift to novel chemotypes) and a random split (the H2 contrast).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# (display name, TDC loader class, TDC dataset name, task, ADME/Tox category)
SELECTED: list[tuple[str, str, str, str, str]] = [
    # Toxicity (all four)
    ("DILI", "Tox", "DILI", "classification", "toxicity"),
    ("hERG", "Tox", "hERG", "classification", "toxicity"),
    ("AMES", "Tox", "AMES", "classification", "toxicity"),
    ("LD50", "Tox", "LD50_Zhu", "regression", "toxicity"),
    # Absorption (reg + class)
    ("Caco2", "ADME", "Caco2_Wang", "regression", "absorption"),
    ("HIA", "ADME", "HIA_Hou", "classification", "absorption"),
    # Distribution (reg + class)
    ("BBB", "ADME", "BBB_Martins", "classification", "distribution"),
    ("VDss", "ADME", "VDss_Lombardo", "regression", "distribution"),
    # Metabolism (classification only in TDC ADMET)
    ("CYP2C9-Sub", "ADME", "CYP2C9_Substrate_CarbonMangels", "classification", "metabolism"),
    ("CYP3A4-Sub", "ADME", "CYP3A4_Substrate_CarbonMangels", "classification", "metabolism"),
    # Excretion (regression only in TDC ADMET)
    ("HalfLife", "ADME", "Half_Life_Obach", "regression", "excretion"),
    ("CL-Hepa", "ADME", "Clearance_Hepatocyte_AZ", "regression", "excretion"),
]


@dataclass
class Endpoint:
    name: str
    tdc_name: str
    task: str
    category: str
    # splits[method] -> {"train": df, "test": df}; each df has columns Drug, Y
    splits: dict


def _loader(cls_name: str):
    if cls_name == "Tox":
        from tdc.single_pred import Tox

        return Tox
    from tdc.single_pred import ADME

    return ADME


def load_endpoint(display: str, cls_name: str, tdc_name: str, task: str, category: str,
                  seed: int = 0) -> Endpoint:
    Loader = _loader(cls_name)
    data = Loader(name=tdc_name)
    splits = {}
    for method in ("scaffold", "random"):
        s = data.get_split(method=method, seed=seed)
        # fit on train (we do not tune heavily); evaluate on test. valid folded into train.
        train = pd.concat([s["train"], s["valid"]], ignore_index=True)[["Drug", "Y"]]
        test = s["test"][["Drug", "Y"]]
        splits[method] = {"train": train, "test": test}
    return Endpoint(display, tdc_name, task, category, splits)


def load_selected(seed: int = 0):
    for display, cls_name, tdc_name, task, category in SELECTED:
        yield load_endpoint(display, cls_name, tdc_name, task, category, seed=seed)
