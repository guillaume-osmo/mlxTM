"""QSAR example: ECFP fingerprints -> Coalesced Tsetlin Machine.

Requires the `examples` extra (rdkit, scikit-learn):  pip install "mlx-tm[examples]"
Expects a CSV with columns SMILES,label.  Usage:  python qsar_ecfp.py data.csv
"""
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

from mlx_tm import CoalescedTsetlinMachine


def ecfp(smiles, n_bits=2048, radius=2):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows, keep = [], np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        rows.append(gen.GetFingerprintAsNumPy(m).astype(np.int8))
        keep[i] = True
    return np.vstack(rows), keep


def main(path):
    df = pd.read_csv(path)
    X, keep = ecfp(df["SMILES"].tolist())
    y = df["label"].to_numpy()[keep].astype(np.int32)
    Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = CoalescedTsetlinMachine(n_clauses=400, T=400, s=10.0, max_weight=64,
                                  weighted_clauses=True, seed=0)
    clf.fit(Xtr, Ytr, epochs=20)
    cs = clf.decision_function(Xte)
    print(f"ROC-AUC: {roc_auc_score(Yte, cs[:, 1] - cs[:, 0]):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
