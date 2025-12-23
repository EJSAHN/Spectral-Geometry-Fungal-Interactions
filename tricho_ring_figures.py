# -*- coding: utf-8 -*-
"""
Trichoderma–Pathogen Ring Geometry (Fig 9) + Mechanism (Fig 10)
==============================================================

This script reproduces the minimal "ring analysis" figures from the VNIR reflectance (Ref) sheet
in `average_data.xlsx`:

- Fig 9: PCA geometry (PC1/PC2) showing Trichoderma, Pathogen, and Ring (interaction) + mixing axis
- Fig 10: Mechanism curves (signed) in spectral space:
          - Collapse targets: Orthogonal vector (Ortho_Vec) highlighted at the water band (~950–990 nm)
          - Resist targets:   NNLS residual highlighted at the blue/pigment-associated band (~430–470 nm)

Outputs (PNG 300 dpi + PDF):
- Fig9_Geometry.png / .pdf
- Fig10_Mechanism.png / .pdf
- Strict_Metrics.csv (per-target summary)

Usage (Anaconda Prompt):
    cd <project_root>
    python tricho_ring_figures.py --excel "Fungus/average_data.xlsx" --out "outputs/paper_final"

Optional:
    python tricho_ring_figures.py --sheet Ref
    python tricho_ring_figures.py --targets "11C-65-1,P24-83,Collapse;11C-65-1,P24-192,Resist"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import nnls
from sklearn.decomposition import PCA


DEFAULT_TARGETS = [
    ("11C-65-1", "P24-83", "Collapse"),
    ("11C-65-1", "P24-192", "Resist"),
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_ref_sheet(xls: pd.ExcelFile, preferred: str | None) -> str:
    if preferred:
        if preferred in xls.sheet_names:
            return preferred
        for s in xls.sheet_names:
            if s.lower() == preferred.lower():
                return s
        raise ValueError(f"Requested sheet '{preferred}' not found. Available: {xls.sheet_names}")
    for s in xls.sheet_names:
        if "ref" in s.lower():
            return s
    raise ValueError(f"No sheet containing 'Ref' found. Available: {xls.sheet_names}")


def load_ref_data(excel_path: Path, sheet: str | None = None) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    sheet_name = _pick_ref_sheet(xls, sheet)

    df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    cols = []
    for c in df.columns:
        try:
            cols.append(float(c))
        except Exception:
            cols.append(np.nan)
    df.columns = cols
    df = df.loc[:, df.columns.notna()].sort_index(axis=1)

    df.index = df.index.astype(str).str.strip()
    return df


def preprocess_reflectance(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.nanmax(X) > 2.0:
            X = X / 100.0
        X = -np.log10(np.clip(X, 1e-6, 1.0))

    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    X = (X - mu) / (sd + 1e-8)

    X = savgol_filter(X, window_length=15, polyorder=3, axis=1)
    return X


@dataclass(frozen=True)
class StrictIDs:
    T_ids: List[str]
    P_ids: List[str]
    R_ids: List[str]


def strict_ids(all_ids: List[str], t_name: str, p_name: str) -> StrictIDs:
    t_re = re.compile(rf"^{re.escape(t_name)}_[1-4]$", flags=re.IGNORECASE)
    p_re = re.compile(rf"^{re.escape(p_name)}_[1-4]$", flags=re.IGNORECASE)
    r_re = re.compile(rf"^{re.escape(p_name)}_{re.escape(t_name)}_ring_[1-4]$", flags=re.IGNORECASE)

    T_ids = [s for s in all_ids if t_re.match(s)]
    P_ids = [s for s in all_ids if p_re.match(s)]
    R_ids = [s for s in all_ids if r_re.match(s)]

    def rep_num(s: str) -> int:
        m = re.search(r"_(\d+)$", s)
        return int(m.group(1)) if m else 999

    def ring_rep_num(s: str) -> int:
        m = re.search(r"_ring_(\d+)$", s, flags=re.IGNORECASE)
        return int(m.group(1)) if m else 999

    T_ids = sorted(T_ids, key=rep_num)
    P_ids = sorted(P_ids, key=rep_num)
    R_ids = sorted(R_ids, key=ring_rep_num)

    return StrictIDs(T_ids=T_ids, P_ids=P_ids, R_ids=R_ids)


def parse_targets(s: str | None) -> List[Tuple[str, str, str]]:
    if not s:
        return list(DEFAULT_TARGETS)
    out: List[Tuple[str, str, str]] = []
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        parts = [p.strip() for p in block.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad target block: '{block}'. Expected T,P,Type")
        out.append((parts[0], parts[1], parts[2]))
    return out


def ortho_ratio_spectral(T_sp: np.ndarray, P_sp: np.ndarray, R_sp: np.ndarray) -> float:
    base = P_sp - T_sp
    v = R_sp - T_sp
    denom = float(np.dot(base, base))
    if denom <= 0:
        return float("nan")
    proj = (np.dot(v, base) / denom) * base
    ortho = v - proj
    return float(np.linalg.norm(ortho) / (np.linalg.norm(base) + 1e-12))


def peak_in_band_signed(y: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> Tuple[float, float]:
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (wl >= lo) & (wl <= hi)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return float("nan"), float("nan")
    sub = np.abs(y[idxs])
    j = int(np.nanargmax(sub))
    k = int(idxs[j])
    return float(wl[k]), float(y[k])


def make_figures(df_raw: pd.DataFrame, out_dir: Path, targets: List[Tuple[str, str, str]]) -> None:
    ensure_dir(out_dir)

    wl = df_raw.columns.to_numpy(dtype=float)
    X_proc = preprocess_reflectance(df_raw.values)

    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(X_proc)
    pca_df = pd.DataFrame(scores, columns=["PC1", "PC2"], index=df_raw.index)

    X_df = pd.DataFrame(X_proc, index=df_raw.index, columns=wl)

    n = len(targets)
    fig9, axs9 = plt.subplots(1, n, figsize=(6.2 * n, 5.4), squeeze=False)
    fig10, axs10 = plt.subplots(1, n, figsize=(6.2 * n, 5.4), squeeze=False)

    metrics_rows = []

    for i, (t_name, p_name, typ) in enumerate(targets):
        ids = strict_ids(list(df_raw.index), t_name=t_name, p_name=p_name)

        print(f"\n[STRICT AUDIT] {t_name} vs {p_name}")
        print(f"  Trichoderma (expected 4): {len(ids.T_ids)} -> {ids.T_ids}")
        print(f"  Pathogen    (expected 4): {len(ids.P_ids)} -> {ids.P_ids}")
        print(f"  Ring        (expected 4): {len(ids.R_ids)} -> {ids.R_ids}")

        if not (len(ids.T_ids) == len(ids.P_ids) == len(ids.R_ids) == 4):
            raise ValueError(
                f"Strict 4 vs 4 vs 4 failed for {t_name} vs {p_name}. "
                f"Found: T={len(ids.T_ids)}, P={len(ids.P_ids)}, R={len(ids.R_ids)}. "
                f"Fix naming or regex in strict_ids()."
            )

        # ---- Fig 9 (PCA geometry)
        ax9 = axs9[0, i]
        role_map = {"Trichoderma": ids.T_ids, "Pathogen": ids.P_ids, "Ring": ids.R_ids}
        colors = {"Trichoderma": "#2ca02c", "Pathogen": "#1f77b4", "Ring": "#d62728"}
        markers = {"Trichoderma": "o", "Pathogen": "o", "Ring": "D"}

        centroids_pc = {}
        for role, id_list in role_map.items():
            pts = pca_df.loc[id_list, ["PC1", "PC2"]].to_numpy()
            ax9.scatter(
                pts[:, 0], pts[:, 1],
                c=colors[role], marker=markers[role],
                s=65, alpha=0.8, edgecolors="white", linewidths=0.7,
                label=role
            )
            centroids_pc[role] = pts.mean(axis=0)

        T_pc = centroids_pc["Trichoderma"]
        P_pc = centroids_pc["Pathogen"]
        R_pc = centroids_pc["Ring"]

        ax9.plot([T_pc[0], P_pc[0]], [T_pc[1], P_pc[1]], "k--", alpha=0.45, lw=1.5, label="Mixing axis")
        VP = P_pc - T_pc
        VR = R_pc - T_pc
        proj = np.dot(VR, VP) / (np.dot(VP, VP) + 1e-12)
        proj_pt = T_pc + proj * VP

        ax9.annotate("", xy=(R_pc[0], R_pc[1]), xytext=(proj_pt[0], proj_pt[1]),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2.0))

        ax9.set_title(f"{p_name}: {typ}", fontweight="bold")
        ax9.set_xlabel("PC1")
        ax9.set_ylabel("PC2")
        ax9.grid(True, alpha=0.25)
        if i == 0:
            ax9.legend(frameon=False, fontsize=10)

        # ---- Fig 10 (spectral mechanism)
        ax10 = axs10[0, i]

        T_sp = X_df.loc[ids.T_ids].mean(axis=0).to_numpy()
        P_sp = X_df.loc[ids.P_ids].mean(axis=0).to_numpy()
        R_sp = X_df.loc[ids.R_ids].mean(axis=0).to_numpy()

        # NNLS residual
        A = np.vstack([T_sp, P_sp]).T
        coef, _ = nnls(A, R_sp)
        residual = R_sp - (A @ coef)

        # Orthogonal vector in spectral space
        base = P_sp - T_sp
        v = R_sp - T_sp
        proj_sp = (np.dot(v, base) / (np.dot(base, base) + 1e-12)) * base
        ortho_vec = v - proj_sp

        if typ.lower().startswith("collapse"):
            y = ortho_vec
            color = "purple"
            band_lo, band_hi = 900, 1000
            span_lo, span_hi = 950, 990
            span_label = "Water-associated band"
            curve_label = "Orthogonal vector"
            subtitle = "Water-band excursion (~980 nm)"
            span_color = "dodgerblue"
        else:
            y = residual
            color = "orange"
            band_lo, band_hi = 420, 550
            span_lo, span_hi = 430, 470
            span_label = "Blue/pigment-associated band"
            curve_label = "NNLS residual"
            subtitle = "Blue-band excursion (~450 nm)"
            span_color = "gold"

        ax10.plot(wl, y, color=color, lw=2.0, label=curve_label)
        ax10.fill_between(wl, 0, y, color=color, alpha=0.12)
        ax10.axhline(0, color="gray", lw=0.8)

        peak_wl, peak_val = peak_in_band_signed(y, wl, band_lo, band_hi)
        ax10.plot(peak_wl, peak_val, "o", color=color, markeredgecolor="black", zorder=5)
        ax10.annotate(
            f"{peak_wl:.0f} nm",
            xy=(peak_wl, peak_val),
            xytext=(peak_wl, peak_val + (0.10 * (1 if peak_val >= 0 else -1))),
            ha="center",
            fontsize=10,
            arrowprops=dict(arrowstyle="-", color="black", lw=1.0),
        )

        ax10.axvspan(span_lo, span_hi, color=span_color, alpha=0.15, label=span_label)
        ax10.set_xlim(float(wl.min()), float(wl.max()))
        ax10.set_xlabel("Wavelength (nm)")
        ax10.set_title(f"{p_name}: {subtitle}", fontweight="bold")
        ax10.grid(True, alpha=0.25)
        if i == n - 1:
            ax10.legend(frameon=False, fontsize=10)

        o_ratio = ortho_ratio_spectral(T_sp, P_sp, R_sp)
        metrics_rows.append({
            "Trichoderma": t_name,
            "Pathogen": p_name,
            "Type": typ,
            "Ortho_Ratio_Spectral": o_ratio,
            "Fig10_Curve": ("Ortho_Vec" if typ.lower().startswith("collapse") else "NNLS_Residual"),
            "Peak_WL": peak_wl,
            "Peak_Value_Signed": peak_val,
            "NNLS_coef_T": float(coef[0]),
            "NNLS_coef_P": float(coef[1]),
        })

    fig9.tight_layout()
    fig10.tight_layout()

    fig9.savefig(out_dir / "Fig9_Geometry.png", dpi=300, bbox_inches="tight")
    fig9.savefig(out_dir / "Fig9_Geometry.pdf", bbox_inches="tight")
    fig10.savefig(out_dir / "Fig10_Mechanism.png", dpi=300, bbox_inches="tight")
    fig10.savefig(out_dir / "Fig10_Mechanism.pdf", bbox_inches="tight")

    pd.DataFrame(metrics_rows).to_csv(out_dir / "Strict_Metrics.csv", index=False)

    print(f"\nSaved outputs to:\n  {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Fig 9 and Fig 10 from strict ring analysis (Ref sheet).")
    ap.add_argument("--excel", type=str, default="Fungus/average_data.xlsx",
                    help="Path to average_data.xlsx (relative to current dir is OK)")
    ap.add_argument("--out", type=str, default="outputs/paper_final",
                    help="Output directory (created if missing)")
    ap.add_argument("--sheet", type=str, default=None,
                    help="Optional sheet name for reflectance (e.g., Ref). Default: first sheet containing 'Ref'.")
    ap.add_argument("--targets", type=str, default=None,
                    help='Targets string: "T,P,Type;T,P,Type". Default uses 11C-65-1 vs P24-83/192.')
    args = ap.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    targets = parse_targets(args.targets)

    df_raw = load_ref_data(excel_path, sheet=args.sheet)
    make_figures(df_raw=df_raw, out_dir=out_dir, targets=targets)


if __name__ == "__main__":
    main()
