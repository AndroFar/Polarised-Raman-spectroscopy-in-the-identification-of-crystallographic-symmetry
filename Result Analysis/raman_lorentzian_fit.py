"""
Raman Spectra Lorentzian Fitting  –  folder batch mode
=======================================================
Processes every .txt file in a folder (or a single file), fits a Lorentzian
to the main peak, and produces an Intensity (A) vs. Angle plot at the end.
 
  L(x) = A · γ² / ((x − x₀)² + γ²)
    A  = peak height above baseline  (directly readable)
    x₀ = peak centre
    γ  = HWHM  →  FWHM = 2γ
 
Angle extraction from filename (in order of priority):
  NNNgraus | NNNdegrees | NNNdeg | trailing NNN after separator
 
Workflow per file:
  1. Read .txt, parse #-metadata, fix comma decimals
  2. Auto-detect linear baseline from spectral edges
  3. Interactive baseline editor  ← make sure spans avoid the peak shoulders!
  4. Subtract baseline, fit Lorentzian
  5. Save individual fit plot + result .txt
 
After all files:
  6. Save fit_summary.csv  (always, even for a single file)
  7. Save A vs. angle plot (if more than one file)
 
Usage:
  python raman_lorentzian_fit.py  <folder>     # batch
  python raman_lorentzian_fit.py  <file.txt>   # single
 
Options:
  --output  DIR    folder for output files  (default: <input>/results)
  --col-x   INT    0-indexed x column       (default: 0)
  --col-y   INT    0-indexed y column       (default: 1)
  --skip-baseline  reuse baseline regions from the first file for all others
"""
 
import sys
import os
import re
import glob
import csv
import argparse
import textwrap
from datetime import datetime
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
from scipy.optimize import curve_fit
 
 
# ── Lorentzian ────────────────────────────────────────────────────────────────
def lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)
 
 
# ── Metadata ──────────────────────────────────────────────────────────────────
def parse_metadata(lines):
    meta = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("#"):
            break
        body = line[1:]
        if "=" in body:
            k, _, v = body.partition("=")
            meta[k.strip()] = v.strip()
    return meta
 
 
def meta_float(meta, key, fallback=None):
    val = meta.get(key, "")
    if not val:
        return fallback
    try:
        return float(val.replace(",", "."))
    except ValueError:
        return fallback
 
 
# ── Angle from filename ───────────────────────────────────────────────────────
def extract_angle(stem):
    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*graus",
        r"(\d+(?:[.,]\d+)?)\s*degrees",
        r"(\d+(?:[.,]\d+)?)\s*deg(?!rees)",
        r"[-_](\d+(?:[.,]\d+)?)(?:[^0-9]|$)",
        r"(\d+(?:[.,]\d+)?)$",
    ]
    for pat in patterns:
        m = re.search(pat, stem, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", "."))
    return None
 
 
# ── File loader ───────────────────────────────────────────────────────────────
def load_spectrum(path, col_x=0, col_y=1):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw_lines = fh.readlines()
    meta = parse_metadata(raw_lines)
    xs, ys = [], []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line  = re.sub(r"(\d),(\d)", r"\1.\2", line)
        parts = re.split(r"[\t;]+|\s+", line)
        parts = [p for p in parts if p]
        try:
            xs.append(float(parts[col_x]))
            ys.append(float(parts[col_y]))
        except (ValueError, IndexError):
            continue
    if not xs:
        raise ValueError(f"No numeric data found in {path}.")
    x = np.array(xs);  y = np.array(ys)
    order = np.argsort(x)
    return x[order], y[order], meta
 
 
# ── Baseline ──────────────────────────────────────────────────────────────────
def auto_baseline_regions(x, y, edge_fraction=0.10):
    span = x[-1] - x[0]
    mask = (x <= x[0] + edge_fraction * span) | (x >= x[-1] - edge_fraction * span)
    return x[mask], y[mask]
 
 
def fit_linear_baseline(x_pts, y_pts):
    return np.poly1d(np.polyfit(x_pts, y_pts, 1))
 
 
# ── Peak estimator ────────────────────────────────────────────────────────────
def estimate_peak(x, y_corr):
    idx_max  = int(np.argmax(y_corr))
    A_guess  = float(y_corr[idx_max])
    x0_guess = float(x[idx_max])
    half = A_guess / 2.0
    li, ri = idx_max, idx_max
    for i in range(idx_max, -1, -1):
        if y_corr[i] < half: li = i; break
    for i in range(idx_max, len(y_corr)):
        if y_corr[i] < half: ri = i; break
    gamma_guess = max((x[ri] - x[li]) / 2.0, (x[1] - x[0]) * 3)
    return A_guess, x0_guess, gamma_guess
 
 
# ── Interactive baseline editor ───────────────────────────────────────────────
class BaselineEditor:
    """
    The two orange shaded spans define which points are used to fit the
    linear baseline.  Keep them well away from the peak shoulders —
    the corrected spectrum should sit at ~0 Cnt in the flat regions.
    """
    def __init__(self, x, y, x_base_auto, filename=""):
        self.x = x;  self.y = y;  self.accepted = False
 
        # Split auto-baseline into left and right halves
        mid       = np.median(x_base_auto)
        left_pts  = x_base_auto[x_base_auto <= mid]
        right_pts = x_base_auto[x_base_auto >  mid]
        self.lx = (float(left_pts.min()),  float(left_pts.max()))
        self.rx = (float(right_pts.min()), float(right_pts.max()))
 
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.fig.subplots_adjust(bottom=0.24, left=0.09, right=0.97, top=0.88)
        self.fig.canvas.manager.set_window_title(
            f"Baseline Editor  –  {filename}"
        )
        self.ax.plot(x, y, color="#3a7ebf", lw=1.1, alpha=0.85,
                     label="Raw spectrum")
 
        x_fine = np.linspace(x[0], x[-1], 1000)
        self.bl_line, = self.ax.plot(
            x_fine, fit_linear_baseline(*self._pts())(x_fine),
            color="#e05c35", lw=1.8, ls="--", label="Baseline"
        )
        self.l_span = self.ax.axvspan(*self.lx, alpha=0.22,
                                       color="#f0a500", label="Baseline regions")
        self.r_span = self.ax.axvspan(*self.rx, alpha=0.22, color="#f0a500")
 
        self.ax.set_xlabel("Raman shift (cm\u207b\u00b9)", fontsize=11)
        self.ax.set_ylabel("Intensity (Cnt)", fontsize=11)
        self.ax.set_title(
            f"{filename}\n"
            "Drag LEFT span and RIGHT span onto the FLAT baseline regions "
            "(avoid peak shoulders!), then click  \u2714 Accept",
            fontsize=10
        )
        self.ax.legend(fontsize=9, loc="upper left")
 
        # Baseline preview text
        self._preview_txt = self.ax.text(
            0.01, 0.02, self._preview_str(),
            transform=self.ax.transAxes, fontsize=8.5,
            family="monospace", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8)
        )
 
        self.sel_l = SpanSelector(
            self.ax, self._on_left,  "horizontal", useblit=True,
            interactive=True, drag_from_anywhere=True,
            props=dict(alpha=0.35, facecolor="#f0a500")
        )
        self.sel_r = SpanSelector(
            self.ax, self._on_right, "horizontal", useblit=True,
            interactive=True, drag_from_anywhere=True,
            props=dict(alpha=0.35, facecolor="#f0a500")
        )
 
        ax_btn = self.fig.add_axes([0.43, 0.05, 0.14, 0.09])
        self.btn = Button(ax_btn, "\u2714  Accept",
                          color="#4caf50", hovercolor="#66bb6a")
        self.btn.label.set_fontsize(11)
        self.btn.on_clicked(self._on_accept)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
 
    def _pts(self):
        mask = (((self.x >= self.lx[0]) & (self.x <= self.lx[1])) |
                ((self.x >= self.rx[0]) & (self.x <= self.rx[1])))
        return self.x[mask], self.y[mask]
 
    def _preview_str(self):
        xp, yp = self._pts()
        if len(xp) < 2:
            return "Not enough baseline points"
        line = fit_linear_baseline(xp, yp)
        mid  = float(np.mean(self.x))
        return (f"Baseline at x={mid:.0f}: {line(mid):.1f} Cnt   "
                f"slope: {line.coefficients[0]:.4f} Cnt/cm\u207b\u00b9   "
                f"n_points: {len(xp)}")
 
    def _redraw(self):
        xp, yp = self._pts()
        if len(xp) < 2: return
        x_fine = np.linspace(self.x[0], self.x[-1], 1000)
        self.bl_line.set_ydata(fit_linear_baseline(xp, yp)(x_fine))
        self.l_span.remove();  self.r_span.remove()
        self.l_span = self.ax.axvspan(*self.lx, alpha=0.22, color="#f0a500")
        self.r_span = self.ax.axvspan(*self.rx, alpha=0.22, color="#f0a500")
        self._preview_txt.set_text(self._preview_str())
        self.fig.canvas.draw_idle()
 
    def _on_left(self,  a, b): self.lx = (a, b); self._redraw()
    def _on_right(self, a, b): self.rx = (a, b); self._redraw()
    def _on_accept(self, _):   self.accepted = True; plt.close(self.fig)
    def _on_close(self, _):    pass
 
    def run(self):
        plt.show(block=True)
        if not self.accepted:
            print("  Baseline check cancelled – exiting.")
            sys.exit(0)
        return self.lx, self.rx
 
 
# ── Single-file fit ───────────────────────────────────────────────────────────
def fit_one(path, col_x, col_y, out_dir, baseline_regions=None):
    fname = os.path.basename(path)
    stem  = os.path.splitext(fname)[0]
    print(f"\n  {'─'*54}")
    print(f"  {fname}")
    print(f"  {'─'*54}")
 
    try:
        x, y, meta = load_spectrum(path, col_x, col_y)
    except ValueError as e:
        print(f"  SKIP – {e}");  return None
 
    title     = meta.get("Title", stem)
    laser_nm  = meta_float(meta, "Laser")
    spec_res  = meta_float(meta, "Spectral res.(cm-\u00b9)")
    acq_time  = meta_float(meta, "Acq. time (s)")
    accum     = meta_float(meta, "Accumulations")
    date_str  = meta.get("Date", "")
    grating   = meta.get("Grating", "")
    objective = meta.get("Objective", "")
 
    # Baseline
    x_base_auto, _ = auto_baseline_regions(x, y)
 
    if baseline_regions is None or "lx" not in baseline_regions:
        # No saved regions yet — open the editor
        editor = BaselineEditor(x, y, x_base_auto, filename=fname)
        lx, rx = editor.run()
        # If a dict was passed in, store the chosen regions for reuse
        if baseline_regions is not None:
            baseline_regions["lx"] = lx
            baseline_regions["rx"] = rx
    else:
        # Reuse previously accepted regions (--skip-baseline mode)
        lx = baseline_regions["lx"]
        rx = baseline_regions["rx"]
        print(f"  Reusing baseline regions: left {lx}, right {rx}")
 
    mask = (((x >= lx[0]) & (x <= lx[1])) |
            ((x >= rx[0]) & (x <= rx[1])))
    if mask.sum() < 2:
        print("  SKIP – baseline regions contain no points.");  return None
 
    line   = fit_linear_baseline(x[mask], y[mask])
    y_corr = y - line(x)
 
    # Check: warn if baseline-corrected flat regions are far from zero
    flat_mask   = mask  # same points used for baseline
    flat_mean   = float(np.mean(y_corr[flat_mask]))
    flat_std    = float(np.std(y_corr[flat_mask]))
    if abs(flat_mean) > 20:
        print(f"\n  *** WARNING: baseline-corrected flat regions sit at "
              f"{flat_mean:.1f} ± {flat_std:.1f} Cnt (should be ~0). ***")
        print(f"  *** The baseline spans may overlap the peak shoulders.  ***\n")
 
    # Fit
    A0, x0_0, g0 = estimate_peak(x, y_corr)
    try:
        popt, pcov = curve_fit(
            lorentzian, x, y_corr,
            p0=[A0, x0_0, g0],
            bounds=([0, x[0], 0], [np.inf, x[-1], np.inf]),
            maxfev=20_000
        )
    except RuntimeError as e:
        print(f"  SKIP – fit failed: {e}");  return None
 
    A_fit, x0_fit, gamma_fit = popt
    perr  = np.sqrt(np.diag(pcov))
    fwhm  = 2.0 * abs(gamma_fit)
    resid = y_corr - lorentzian(x, *popt)
    rms   = float(np.sqrt(np.mean(resid**2)))
 
    print(f"  x0   = {x0_fit:.4f}  +/-  {perr[1]:.4f}  cm-1")
    print(f"  A    = {A_fit:.2f}   +/-  {perr[0]:.2f}   Cnt")
    print(f"  FWHM = {fwhm:.4f}  cm-1"
          + (f"  (res {spec_res:.1f}, ratio {fwhm/spec_res:.1f}x)" if spec_res else ""))
    print(f"  Baseline at peak: {line(x0_fit):.1f} Cnt  "
          f"(flat region mean after corr: {flat_mean:.1f} Cnt)")
    print(f"  RMS  = {rms:.3f}  Cnt")
 
    # ── Plot ─────────────────────────────────────────────────────────────────
    x_fine = np.linspace(x[0], x[-1], 3000)
    y_fine = lorentzian(x_fine, *popt)
 
    fig, (ax_m, ax_r) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, left=0.11, right=0.97, top=0.91, bottom=0.10)
 
    ax_m.plot(x, y_corr, "o", ms=3.5, color="#3a7ebf",
              alpha=0.55, zorder=2, label="Baseline-corrected")
    ax_m.plot(x_fine, y_fine, color="#e05c35", lw=2.2,
              zorder=3, label="Lorentzian fit")
    ax_m.axhline(0, color="#aaa", lw=0.8, ls=":")   # zero reference
 
    y_half = A_fit / 2.0
    xl, xr = x0_fit - abs(gamma_fit), x0_fit + abs(gamma_fit)
    ax_m.hlines(y_half, xl, xr, colors="#666", lw=1.2, ls=":", zorder=4)
    ax_m.annotate("", xy=(xr, y_half), xytext=(xl, y_half),
                  arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3))
    ax_m.text(x0_fit, y_half + A_fit * 0.03,
              f"FWHM = {fwhm:.3f} cm\u207b\u00b9",
              ha="center", va="bottom", fontsize=9, color="#333")
    ax_m.annotate("", xy=(x0_fit, A_fit), xytext=(x0_fit, 0),
                  arrowprops=dict(arrowstyle="<->", color="#888", lw=1.1, ls="dashed"))
    ax_m.text(x0_fit + (x[-1]-x[0])*0.012, A_fit * 0.52,
              f"A = {A_fit:.1f} Cnt", va="center", fontsize=9, color="#333")
 
    ax_m.set_ylabel("Intensity (Cnt)", fontsize=11)
    ax_m.legend(fontsize=9, loc="upper left")
    ax_m.tick_params(labelbottom=False)
 
    sub = "   |   ".join(filter(None, [
        f"\u03bb={laser_nm:.0f} nm" if laser_nm else "",
        f"grating {grating}"        if grating   else "",
        f"obj. {objective}"         if objective  else "",
        date_str
    ]))
    fig.suptitle(f"{title}\n{sub}", fontsize=10, y=0.97)
 
    box = (f"x\u2080 = {x0_fit:.4f} \u00b1 {perr[1]:.4f} cm\u207b\u00b9\n"
           f"A  = {A_fit:.2f} \u00b1 {perr[0]:.2f} Cnt\n"
           f"FWHM = {fwhm:.4f} cm\u207b\u00b9")
    if spec_res:
        box += f"\nFWHM/res = {fwhm/spec_res:.1f}\u00d7"
    ax_m.text(0.975, 0.97, box, transform=ax_m.transAxes,
              ha="right", va="top", fontsize=8.5, family="monospace",
              bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#aaa", alpha=0.85))
 
    ax_r.axhline(0, color="#888", lw=0.9, ls="--")
    ax_r.plot(x, resid, ".", ms=3, color="#5a7fa8", alpha=0.65)
    ax_r.set_xlabel("Raman shift (cm\u207b\u00b9)", fontsize=11)
    ax_r.set_ylabel("Residuals\n(Cnt)", fontsize=8.5)
 
    out_png = os.path.join(out_dir, stem + "_lorentzian_fit.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  ->  {out_png}")
 
    # ── Results .txt ──────────────────────────────────────────────────────────
    out_txt = os.path.join(out_dir, stem + "_fit_results.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("Raman Lorentzian Fit Results\n" + "="*45 + "\n")
        fh.write(f"Analysed on    : {now}\n")
        fh.write(f"Input file     : {path}\n")
        if title:     fh.write(f"Title          : {title}\n")
        if date_str:  fh.write(f"Acquired       : {date_str}\n")
        if laser_nm:  fh.write(f"Laser          : {laser_nm:.2f} nm\n")
        if grating:   fh.write(f"Grating        : {grating}\n")
        if objective: fh.write(f"Objective      : {objective}\n")
        if acq_time:  fh.write(f"Acq. time      : {acq_time:.0f} s "
                                f"x {int(accum) if accum else '?'} accum.\n")
        fh.write("-"*45 + "\n")
        fh.write(f"Peak position  : {x0_fit:.6f}  +/-  {perr[1]:.6f}  cm-1\n")
        fh.write(f"Peak height    : {A_fit:.4f}  +/-  {perr[0]:.4f}  Cnt\n")
        fh.write(f"HWHM (gamma)   : {abs(gamma_fit):.6f}  +/-  {perr[2]:.6f}  cm-1\n")
        fh.write(f"FWHM  (2*gamma): {fwhm:.6f}  cm-1\n")
        if spec_res:
            fh.write(f"Instr. res.    : {spec_res:.1f}  cm-1\n")
            fh.write(f"FWHM / res.    : {fwhm/spec_res:.2f}x\n")
        fh.write(f"Residual RMS   : {rms:.4f}  Cnt\n")
        fh.write(f"Baseline check : flat region mean after corr = {flat_mean:.1f} Cnt "
                 f"(target: ~0)\n")
        fh.write("-"*45 + "\n")
        fh.write(f"Baseline slope : {line.coefficients[0]:.6e}  Cnt/cm-1\n")
        fh.write(f"Baseline intcp.: {line.coefficients[1]:.6e}  Cnt\n")
 
    return {
        "file":      fname,
        "angle":     extract_angle(stem),
        "x0":        x0_fit,        "x0_err":    perr[1],
        "A":         A_fit,         "A_err":     perr[0],
        "gamma":     abs(gamma_fit),"gamma_err": perr[2],
        "fwhm":      fwhm,
        "rms":       rms,
        "flat_mean": flat_mean,
        "spec_res":  spec_res,
    }
 
 
# ── A vs. angle plot ──────────────────────────────────────────────────────────
def plot_intensity_vs_angle(results, out_dir):
    angled = sorted([r for r in results if r["angle"] is not None],
                    key=lambda r: r["angle"])
    if not angled:
        print("\n  No angle information found – skipping A vs. angle plot.")
        return
 
    angles = np.array([r["angle"]  for r in angled])
    As     = np.array([r["A"]      for r in angled])
    A_errs = np.array([r["A_err"]  for r in angled])
    fwhms  = np.array([r["fwhm"]   for r in angled])
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.subplots_adjust(hspace=0.08, left=0.12, right=0.95, top=0.93, bottom=0.10)
 
    ax1.errorbar(angles, As, yerr=A_errs, fmt="o-", color="#3a7ebf",
                 ms=6, lw=1.8, ecolor="#7aafd4", elinewidth=1.5,
                 capsize=4, label="Peak height A")
    ax1.set_ylabel("Peak height  A  (Cnt)", fontsize=11)
    ax1.legend(fontsize=9);  ax1.grid(axis="y", ls="--", alpha=0.4)
    fig.suptitle("Lorentzian Fit Results vs. Angle", fontsize=12, y=0.96)
 
    mean_fwhm = float(np.mean(fwhms))
    ax2.axhline(mean_fwhm, color="#e05c35", lw=1.2, ls="--",
                label=f"Mean FWHM = {mean_fwhm:.3f} cm\u207b\u00b9")
    ax2.plot(angles, fwhms, "s-", color="#e05c35", ms=5, lw=1.5)
    for ang, fw, r in zip(angles, fwhms, angled):
        if abs(fw - mean_fwhm) / mean_fwhm > 0.20:
            ax2.annotate("!", xy=(ang, fw), xytext=(ang, fw + mean_fwhm*0.05),
                         ha="center", color="red", fontsize=11, fontweight="bold")
    ax2.set_ylabel("FWHM  (cm\u207b\u00b9)", fontsize=11)
    ax2.set_xlabel("Angle  (\u00b0)", fontsize=11)
    ax2.legend(fontsize=9);  ax2.grid(axis="y", ls="--", alpha=0.4)
 
    out_png = os.path.join(out_dir, "intensity_vs_angle.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n  A vs. angle plot  ->  {out_png}")
    plt.show(block=True)
    plt.close(fig)
 
 
# ── CSV writer (always called) ────────────────────────────────────────────────
def save_csv(results, out_dir):
    out_csv = os.path.join(out_dir, "fit_summary.csv")
    fields  = ["file", "angle", "x0", "x0_err",
               "A", "A_err", "gamma", "gamma_err",
               "fwhm", "rms", "flat_mean"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  fit_summary.csv   ->  {out_csv}")
    return out_csv
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch Raman Lorentzian fit + Intensity vs Angle plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__)
    )
    parser.add_argument("input",  help="Folder with .txt spectra, or single .txt file")
    parser.add_argument("--output",         default=None)
    parser.add_argument("--col-x",          type=int, default=0)
    parser.add_argument("--col-y",          type=int, default=1)
    parser.add_argument("--skip-baseline",  action="store_true",
                        help="Reuse baseline regions from the first file for all others")
    args = parser.parse_args()
 
    inp = args.input
    if os.path.isfile(inp):
        files   = [inp]
        out_dir = args.output or os.path.join(
                      os.path.dirname(os.path.abspath(inp)), "results")
    elif os.path.isdir(inp):
        files   = sorted(glob.glob(os.path.join(inp, "*.txt")))
        if not files:
            print(f"No .txt files found in {inp}");  sys.exit(1)
        out_dir = args.output or os.path.join(inp, "results")
    else:
        print(f"Error: {inp} is not a file or folder.");  sys.exit(1)
 
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Output folder : {out_dir}")
    print(f"  Files found   : {len(files)}")
 
    results        = []
    baseline_state = {}
 
    for fpath in files:
        # With --skip-baseline: always pass the same dict so fit_one fills it
        # on the first file and reads from it on all subsequent ones.
        # Without --skip-baseline: pass None so the editor opens every time.
        regions = baseline_state if args.skip_baseline else None
        res = fit_one(fpath, args.col_x, args.col_y, out_dir,
                      baseline_regions=regions)
        if res is None:
            continue
        results.append(res)
 
    if not results:
        print("\n  No successful fits.");  sys.exit(1)
 
    print(f"\n  Fitted {len(results)} / {len(files)} spectra successfully.")
 
    # Always save CSV
    csv_path = save_csv(results, out_dir)
 
    if len(results) > 1:
        plot_intensity_vs_angle(results, out_dir)
    else:
        r = results[0]
        print(f"\n  Single file mode – CSV written to {csv_path}")
        print(f"  Run raman_polar.py on it when you have all angle files.")
 
 
if __name__ == "__main__":
    main()