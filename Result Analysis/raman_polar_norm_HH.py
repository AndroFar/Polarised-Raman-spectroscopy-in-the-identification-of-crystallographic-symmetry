"""
Raman Polar Plot  –  HH (parallel polarisation)
================================================
Reads fit_summary.csv, normalises peak height and integrated area to [0,1],
and overlays the theoretical model for HH geometry:
 
  I(theta) = sin²(2*theta)
 
Outputs:
  polar_height_HH.png      – normalised A vs angle + theory
  polar_area_HH.png        – normalised area vs angle + theory
  polar_combined_HH.png    – both side by side
  angle_intensity_table_HH.txt / .csv
 
Usage:
  python raman_polar_HH.py  fit_summary.csv
  python raman_polar_HH.py  fit_summary.csv  --output  my_results/
"""
 
import sys
import os
import csv
import argparse
import textwrap
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
 
GEOMETRY  = "HH"
COLOR_EXP = "#3a7ebf"
OFFSET_DEG = 51.0   # angle between sample axis and stage zero (degrees)
 
 
# ── Area ──────────────────────────────────────────────────────────────────────
def integrated_area(A, gamma):
    return A * gamma * np.pi
 
def area_uncertainty(A, sigma_A, gamma, sigma_gamma):
    return np.pi * np.sqrt((gamma * sigma_A)**2 + (A * sigma_gamma)**2)
 
 
# ── Theory ────────────────────────────────────────────────────────────────────
def theory(angles_deg):
    """HH: I(theta) = sin²(2*(theta - OFFSET_DEG))"""
    return np.sin(2 * np.deg2rad(np.asarray(angles_deg, dtype=float) - OFFSET_DEG)) ** 2
 
 
# ── CSV reader ────────────────────────────────────────────────────────────────
def load_summary(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("bad_fit", "")).strip().lower() in ("true", "1", "yes"):
                print(f"  Skipping bad fit: {row.get('file','')}  "
                      f"angle={row.get('angle','')}  "
                      f"reason: {row.get('bad_fit_reason','')}")
                continue
            try:
                rows.append({
                    "file":      row["file"],
                    "angle":     float(row["angle"]),
                    "A":         float(row["A"]),
                    "A_err":     float(row["A_err"]),
                    "gamma":     float(row["gamma"]),
                    "gamma_err": float(row["gamma_err"]),
                    "fwhm":      float(row["fwhm"]),
                    "x0":        float(row["x0"]),
                    "x0_err":    float(row["x0_err"]),
                    "rms":       float(row["rms"]),
                })
            except (KeyError, ValueError) as e:
                print(f"  Warning: skipping row ({e}): {row}")
    if not rows:
        raise ValueError("No valid rows found in CSV.")
    rows.sort(key=lambda r: r["angle"])
    return rows
 
 
# ── Enrich + normalise ────────────────────────────────────────────────────────
def enrich(rows):
    for r in rows:
        r["area"]     = integrated_area(r["A"], r["gamma"])
        r["area_err"] = area_uncertainty(r["A"], r["A_err"],
                                          r["gamma"], r["gamma_err"])
    A_max    = max(r["A"]    for r in rows)
    area_max = max(r["area"] for r in rows)
    for r in rows:
        r["A_norm"]        = r["A"]        / A_max
        r["A_norm_err"]    = r["A_err"]    / A_max
        r["area_norm"]     = r["area"]     / area_max
        r["area_norm_err"] = r["area_err"] / area_max
    return rows, A_max, area_max
 
 
# ── Table ─────────────────────────────────────────────────────────────────────
def build_table(rows):
    hdr = (f"{'Angle':>7}  {'A (Cnt)':>12}  {'±':>8}  {'A_norm':>8}  "
           f"{'Area (Cnt·cm⁻¹)':>16}  {'±':>10}  {'Area_norm':>10}  {'FWHM':>8}")
    sep   = "─" * len(hdr)
    lines = [sep, hdr, sep]
    for r in rows:
        lines.append(
            f"{r['angle']:>7.1f}  {r['A']:>12.2f}  {r['A_err']:>8.2f}  "
            f"{r['A_norm']:>8.4f}  {r['area']:>16.3f}  {r['area_err']:>10.3f}  "
            f"{r['area_norm']:>10.4f}  {r['fwhm']:>8.4f}"
        )
    lines.append(sep)
    As    = np.array([r["A"]    for r in rows])
    areas = np.array([r["area"] for r in rows])
    fwhms = np.array([r["fwhm"] for r in rows])
    lines += [
        "",
        f"  Peak height A   :  mean {As.mean():.2f}  std {As.std():.2f}  "
        f"min {As.min():.2f}  max {As.max():.2f}  Cnt",
        f"  Integrated area :  mean {areas.mean():.3f}  std {areas.std():.3f}  "
        f"min {areas.min():.3f}  max {areas.max():.3f}  Cnt\u00b7cm\u207b\u00b9",
        f"  FWHM            :  mean {fwhms.mean():.4f}  std {fwhms.std():.4f}  cm\u207b\u00b9",
    ]
    return "\n".join(lines)
 
 
# ── Draw one polar axis ───────────────────────────────────────────────────────
def draw_polar(ax, angles_deg, values_norm, errors_norm, title):
    phi_fine   = np.linspace(0, 360, 1440)
    theta_fine = np.deg2rad(phi_fine)
    I_theory   = theory(phi_fine)
 
    theta_exp = np.deg2rad(angles_deg)
    vals = values_norm.copy()
    errs = errors_norm.copy()
    if not np.isclose(theta_exp[0], theta_exp[-1]):
        theta_exp = np.append(theta_exp, theta_exp[0])
        vals      = np.append(vals,      vals[0])
        errs      = np.append(errs,      errs[0])
 
    ax.plot(theta_fine, I_theory,
            color="#e05c35", lw=1.8, ls="--", zorder=3,
            label=f"Theory: sin\u00b2(2(\u03b8\u2212{OFFSET_DEG:.0f}\u00b0))")
    ax.plot(theta_exp, vals,
            "o-", color=COLOR_EXP, lw=1.6, ms=4, zorder=4,
            label="Experiment (norm.)")
    ax.fill_between(theta_exp,
                    np.maximum(vals - errs, 0), vals + errs,
                    alpha=0.20, color=COLOR_EXP, zorder=2)
 
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=8)
    ax.set_title(title, pad=16, fontsize=10)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.30, 1.10))
 
 
# ── Main plotting ─────────────────────────────────────────────────────────────
def plot_polar(rows, out_dir):
    angles  = np.array([r["angle"]         for r in rows])
    A_norm  = np.array([r["A_norm"]        for r in rows])
    A_nerr  = np.array([r["A_norm_err"]    for r in rows])
    ar_norm = np.array([r["area_norm"]     for r in rows])
    ar_nerr = np.array([r["area_norm_err"] for r in rows])
 
    # Peak height
    fig1, ax1 = plt.subplots(figsize=(6.5, 6.5),
                              subplot_kw={"projection": "polar"})
    fig1.subplots_adjust(top=0.85, right=0.78)
    draw_polar(ax1, angles, A_norm, A_nerr,
               "HH  –  Peak height A  (normalised)")
    out1 = os.path.join(out_dir, "polar_height_HH.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved  ->  {out1}")
 
    # Integrated area
    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5),
                              subplot_kw={"projection": "polar"})
    fig2.subplots_adjust(top=0.85, right=0.78)
    draw_polar(ax2, angles, ar_norm, ar_nerr,
               "HH  –  Integrated area A\u00b7\u03b3\u00b7\u03c0  (normalised)")
    out2 = os.path.join(out_dir, "polar_area_HH.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved  ->  {out2}")
 
    # Combined
    fig3 = plt.figure(figsize=(14, 6.5))
    fig3.subplots_adjust(wspace=0.55, top=0.85, left=0.05, right=0.95)
    draw_polar(fig3.add_subplot(121, projection="polar"),
               angles, A_norm, A_nerr,
               "HH  –  Peak height A  (normalised)")
    draw_polar(fig3.add_subplot(122, projection="polar"),
               angles, ar_norm, ar_nerr,
               "HH  –  Integrated area A\u00b7\u03b3\u00b7\u03c0  (normalised)")
    fig3.suptitle(
        "HH  –  Normalised Raman Intensity vs. Stage Angle \u03b8\n"
        f"Theory: I \u221d sin\u00b2(2(\u03b8\u2212{OFFSET_DEG:.0f}\u00b0))  |  offset = {OFFSET_DEG:.0f}\u00b0",
        fontsize=12, y=0.97
    )
    out3 = os.path.join(out_dir, "polar_combined_HH.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"  Saved  ->  {out3}")
    plt.show(block=True)
    plt.close(fig3)
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="HH polar plots with sin²(2θ) theoretical overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__)
    )
    parser.add_argument("csv",     help="Path to fit_summary.csv")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: same folder as CSV)")
    args = parser.parse_args()
 
    if not os.path.isfile(args.csv):
        print(f"Error: file not found - {args.csv}"); sys.exit(1)
 
    out_dir = args.output or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
 
    print(f"\n  Reading   {args.csv}")
    print(f"  Geometry  HH  |  Theory: sin\u00b2(2(\u03b8-{OFFSET_DEG:.0f}\u00b0))  offset={OFFSET_DEG:.0f}\u00b0")
    rows = load_summary(args.csv)
    print(f"  Good fits {len(rows)}\n")
 
    rows, A_max, area_max = enrich(rows)
    print(f"  Normalisation:  A_max = {A_max:.2f} Cnt   "
          f"area_max = {area_max:.3f} Cnt\u00b7cm\u207b\u00b9")
 
    table_str = build_table(rows)
    print("\n" + table_str)
 
    out_txt = os.path.join(out_dir, "angle_intensity_table_HH.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("Raman Intensity vs. Angle  |  HH geometry\n\n")
        fh.write(table_str + "\n")
    print(f"\n  Table saved  ->  {out_txt}")
 
    out_csv = os.path.join(out_dir, "angle_intensity_table_HH.csv")
    fields = ["angle", "A", "A_err", "A_norm", "A_norm_err",
              "area", "area_err", "area_norm", "area_norm_err",
              "fwhm", "x0", "x0_err", "rms"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"  CSV saved    ->  {out_csv}")
 
    print()
    plot_polar(rows, out_dir)
    print("\n  Done.\n")
 
 
if __name__ == "__main__":
    main()