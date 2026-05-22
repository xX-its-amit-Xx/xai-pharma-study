"""Molecular featurization: RDKit 2D descriptors and Morgan/ECFP4 fingerprints.

Two representations used throughout Paper 1:
- ``descriptors``: ~200 interpretable RDKit 2D descriptors (MolWt, TPSA, logP, ...).
  These are the human-readable features a med-chemist reasons about, and the setting
  where "feature X is important" is most directly actionable.
- ``ecfp``: 2048-bit Morgan/ECFP4 circular fingerprints (radius 2). The de-facto
  standard structural representation; bits correspond to local substructures.

Featurized arrays are cached to ``data/cache/`` keyed by (dataset, split, fold, rep)
so re-runs and the P4 attribution phase reuse identical inputs.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_DESC = [(n, f) for n, f in Descriptors._descList]
DESCRIPTOR_NAMES = [n for n, _ in _DESC]
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ECFP_NAMES = [f"ecfp_{i}" for i in range(2048)]


def feature_names(representation: str) -> list[str]:
    return DESCRIPTOR_NAMES if representation == "descriptors" else ECFP_NAMES


def _descriptors(mols) -> np.ndarray:
    rows = []
    for m in mols:
        vals = []
        for _, fn in _DESC:
            try:
                vals.append(float(fn(m)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    X = np.asarray(rows, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _ecfp(mols) -> np.ndarray:
    arr = np.zeros((len(mols), 2048), dtype=np.float64)
    for i, m in enumerate(mols):
        fp = _MORGAN.GetFingerprintAsNumPy(m)
        arr[i] = fp
    return arr


def _featurize_smiles(smiles: list[str], representation: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, valid_mask). Invalid SMILES are dropped via the mask."""
    mols, mask = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        mask.append(m is not None)
        if m is not None:
            mols.append(m)
    X = _descriptors(mols) if representation == "descriptors" else _ecfp(mols)
    return X, np.asarray(mask)


def _cache_key(dataset: str, split: str, fold: str, representation: str, smiles: list[str]) -> str:
    h = hashlib.md5("".join(smiles).encode()).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"{dataset}__{split}__{fold}__{representation}__{h}.npz")


def featurize(dataset: str, split: str, fold: str, representation: str,
              smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Cached featurization for one (dataset, split, fold, representation).

    ``fold`` is "train"/"test"; ``split`` is "scaffold"/"random". Returns (X, mask)
    aligned to the input SMILES order (mask marks SMILES that RDKit could parse).
    """
    path = _cache_key(dataset, split, fold, representation, smiles)
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["mask"]
    X, mask = _featurize_smiles(smiles, representation)
    np.savez_compressed(path, X=X, mask=mask)
    return X, mask
