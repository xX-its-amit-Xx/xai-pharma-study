"""Paper 3 P12 main experiment.

Scales the P11 feasibility result and tests D1-D3:
  D1 - mean ground-truth-atom-recovery AUROC > 0.6 for the best attribution method.
  D2 - null-referenced faithfulness ordering matches recovery ordering across methods.
  D3 - characterize where recovery fails (bit, molecule properties).

Setup: ~1000 chemically-diverse SMILES; multi-task GIN over top-K=128 Morgan bits;
attribution methods = {graph-IG, atom occlusion, random}; bootstrap CIs over (mol,bit)
pairs. Faithfulness (for D2) = atom-level comprehensiveness: mask top-attributed atoms,
measure drop in the target bit's predicted probability; compare to a random-attribution
null in the same way as Paper 1.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import from_smiles

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(__file__))
import data as data_mod  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
N_BITS = 1024
RADIUS = 2
K_BITS = 128
N_MOLS = 1000
torch.manual_seed(0)


def morgan_with_bitinfo(mol):
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS, bitInfo=bi)
    return np.array(fp, dtype=np.float32), bi


def gt_atoms(bi, bit):
    return {int(a) for a, _ in bi.get(bit, [])}


class MultiBitGIN(nn.Module):
    def __init__(self, in_dim=9, hidden=64, n_out=K_BITS, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        prev = in_dim
        for _ in range(n_layers):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(prev, hidden), nn.ReLU(),
                                                    nn.Linear(hidden, hidden))))
            prev = hidden
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, edge_index, batch):
        h = x.float()
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        return self.head(global_add_pool(h, batch))


def graph_ig(model, x, edge_index, batch, target_bit, n_steps=20):
    baseline = torch.zeros_like(x); grads = []
    for a in torch.linspace(0, 1, n_steps):
        x_a = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        out = model(x_a, edge_index, batch)[0, target_bit]
        grads.append(torch.autograd.grad(out, x_a)[0])
    return (torch.stack(grads).mean(0) * (x - baseline)).detach().numpy()


@torch.no_grad()
def occlusion_scores(model, x, edge_index, batch, target_bit):
    base = torch.sigmoid(model(x, edge_index, batch))[0, target_bit].item()
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        x2 = x.clone(); x2[i] = 0.0
        out[i] = base - torch.sigmoid(model(x2, edge_index, batch))[0, target_bit].item()
    return out


@torch.no_grad()
def comprehensiveness(model, x, edge_index, batch, target_bit, atom_scores, fracs=(0.1, 0.2, 0.3)):
    base = torch.sigmoid(model(x, edge_index, batch))[0, target_bit].item()
    order = np.argsort(-np.abs(atom_scores))
    drops = []
    for f in fracs:
        k = max(1, int(round(f * x.shape[0])))
        x2 = x.clone(); x2[order[:k]] = 0.0
        drops.append(base - torch.sigmoid(model(x2, edge_index, batch))[0, target_bit].item())
    return float(np.mean(drops))


def main():
    # Diverse SMILES pool
    eps = {e.name: e for e in data_mod.load_selected(seed=0)}
    smis = []
    for name in ["AMES", "LD50", "DILI", "BBB", "Caco2"]:
        smis.extend(eps[name].splits["scaffold"]["train"]["Drug"].tolist())
    smis = list(dict.fromkeys(smis))[:N_MOLS]
    items = []
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        g = from_smiles(smi)
        if g.num_nodes == 0 or g.edge_index.numel() == 0: continue
        fp, bi = morgan_with_bitinfo(m)
        items.append((g, fp, bi))
    print(f"molecules: {len(items)}")

    freqs = np.stack([fp for _, fp, _ in items]).sum(0)
    top_bits = np.argsort(-freqs)[:K_BITS].tolist()
    print(f"K={K_BITS} top bits, frequency {int(freqs[top_bits].min())}-{int(freqs[top_bits].max())}/{len(items)}")

    dataset, bis = [], []
    for g, fp, bi in items:
        dataset.append(Data(x=g.x.float(), edge_index=g.edge_index,
                            y=torch.tensor([[fp[b] for b in top_bits]], dtype=torch.float32)))
        bis.append(bi)
    n_tr = int(0.7 * len(dataset))
    tr, te, te_bis = dataset[:n_tr], dataset[n_tr:], bis[n_tr:]

    model = MultiBitGIN(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(); loader = DataLoader(tr, batch_size=64, shuffle=True)
    for _ in range(80):
        model.train()
        for b in loader:
            opt.zero_grad(); loss_fn(model(b.x, b.edge_index, b.batch), b.y).backward(); opt.step()
    model.eval()

    # collect per (mol, bit) records
    rng = np.random.default_rng(0)
    rows = []
    for mi, (d, bi) in enumerate(zip(te, te_bis)):
        batch = torch.zeros(d.num_nodes, dtype=torch.long)
        active = [j for j, b in enumerate(top_bits) if d.y[0, j].item() > 0.5 and b in bi]
        if d.num_nodes < 3 or not active: continue
        for j in active:
            gt = gt_atoms(bi, top_bits[j])
            y_bin = np.array([1 if a in gt else 0 for a in range(d.num_nodes)])
            if not gt or len(np.unique(y_bin)) < 2: continue
            ig_s = np.abs(graph_ig(model, d.x.float(), d.edge_index, batch, j)).sum(axis=1)
            occ_s = occlusion_scores(model, d.x.float(), d.edge_index, batch, j)
            rand_s = rng.standard_normal(d.num_nodes)
            rows.append({
                "mol_idx": mi, "bit": top_bits[j], "n_atoms": d.num_nodes,
                "n_gt": len(gt), "frac_gt": len(gt) / d.num_nodes,
                "ig_auc": roc_auc_score(y_bin, ig_s),
                "occ_auc": roc_auc_score(y_bin, occ_s),
                "rand_auc": roc_auc_score(y_bin, rand_s),
                "ig_faith": comprehensiveness(model, d.x.float(), d.edge_index, batch, j, ig_s),
                "occ_faith": comprehensiveness(model, d.x.float(), d.edge_index, batch, j, occ_s),
                "rand_faith": comprehensiveness(model, d.x.float(), d.edge_index, batch, j, rand_s),
            })
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "p12_main.csv"), index=False)
    n = len(df)

    def boot(vals):
        v = np.asarray(vals); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    print(f"\n=== Paper 3 P12: n={n} (mol, bit) pairs ===")
    print(f"\nD1 - ground-truth-atom recovery (mean AUROC, 95% CI):")
    for k in ["ig_auc", "occ_auc", "rand_auc"]:
        m, lo, hi = boot(df[k]); print(f"  {k:8s} {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    pass_d1 = df.occ_auc.mean() > 0.6 or df.ig_auc.mean() > 0.6
    print(f"D1 {'SUPPORTED' if pass_d1 else 'FALSIFIED'}")

    print(f"\nD2 - faithfulness ordering vs recovery ordering across methods:")
    method_recovery = {"IG": df.ig_auc.mean(), "occlusion": df.occ_auc.mean(), "random": df.rand_auc.mean()}
    method_faith = {"IG": df.ig_faith.mean(), "occlusion": df.occ_faith.mean(), "random": df.rand_faith.mean()}
    print(f"  recovery   : {method_recovery}")
    print(f"  faithfulness: {method_faith}")
    methods = ["IG", "occlusion", "random"]
    rec = [method_recovery[m] for m in methods]; fai = [method_faith[m] for m in methods]
    rho = spearmanr(rec, fai).statistic
    print(f"  Spearman(recovery, faithfulness across 3 methods) = {rho:+.3f}")
    print(f"D2 {'SUPPORTED' if rho > 0.3 else 'NOT supported'} (faithfulness {'tracks' if rho>0.3 else 'does NOT track'} ground-truth recovery)")

    print(f"\nD3 - characterization (Spearman with occlusion recovery):")
    for col in ["n_atoms", "n_gt", "frac_gt"]:
        r = spearmanr(df[col], df.occ_auc).statistic
        print(f"  {col:8s} vs occ_auc: rho = {r:+.3f}")


if __name__ == "__main__":
    main()
