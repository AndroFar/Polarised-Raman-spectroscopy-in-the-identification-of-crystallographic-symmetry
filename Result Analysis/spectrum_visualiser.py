"""
Raman Spectrum Visualiser
=========================
Reads one or two LabRAM / Horiba .txt files and plots them overlaid.

Usage:
  python raman_plot.py  spectrum.txt
  python raman_plot.py  spec1.txt  spec2.txt
  python raman_plot.py  spec1.txt  spec2.txt  --labels "HH" "HV"
  python raman_plot.py  spec1.txt  spec2.txt  --xmin 480 --xmax 560
  python raman_plot.py  spec1.txt  spec2.txt  --save
"""

import sys
import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt

COLORS = ["#3a7ebf", "#e05c35"]


def load(path):
    """Read file, fix comma decimals, extract metadata and data."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.readlines()

    meta = {}
    for line in raw:
        line = line.strip()
        if not line.startswith("#"):
            break
        if "=" in line:
            k, _, v = line[1:].partition("=")
            meta[k.strip()] = v.strip()

    xs, ys = [], []
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"(\d),(\d)", r"\1.\2", line)
        parts = re.split(r"[\t;]+|\s+", line)
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except (ValueError, IndexError):
            continue

    return np.array(xs), np.array(ys), meta


def meta_subtitle(meta):
    """Generates hardware configurations. Timestamp omitted to avoid multi-line redundancy."""
    parts = []
    if meta.get("Laser"):
        parts.append(f"\u03bb={float(meta['Laser'].replace(',','.')):g} nm")
    if meta.get("Grating"):
        parts.append(f"gr.{meta['Grating']}")
    if meta.get("Objective"):
        parts.append(meta["Objective"])
    if meta.get("Acq. time (s)") and meta.get("Accumulations"):
        t = meta["Acq. time (s)"].replace(",", ".")
        n = meta["Accumulations"].replace(",", ".")
        parts.append(f"{t}s\u00d7{n}")
    return "  |  ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Plot one or two Raman spectra from .txt files.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help="One or two spectrum .txt files")
    parser.add_argument("--labels", nargs="+", metavar="LABEL", default=None,
                        help="Legend labels, one per file")
    parser.add_argument("--title",  default=None,
                        help="Plot title (default: filename(s))")
    parser.add_argument("--xmin",   type=float, default=None,
                        help="x-axis minimum (cm-1)")
    parser.add_argument("--xmax",   type=float, default=None,
                        help="x-axis maximum (cm-1)")
    parser.add_argument("--save",   action="store_true",
                        help="Save plot as PNG instead of showing it")
    args = parser.parse_args()

    if len(args.files) > 2:
        print("Error: at most 2 files can be overlaid."); sys.exit(1)

    for f in args.files:
        if not os.path.isfile(f):
            print(f"Error: file not found - {f}"); sys.exit(1)

    datasets = [load(f) for f in args.files]

    # Default labels: filename stem or user-supplied
    if args.labels and len(args.labels) >= len(args.files):
        labels = args.labels[:len(args.files)]
    else:
        labels = [os.path.splitext(os.path.basename(f))[0]
                  for f in args.files]

    # ── Custom Title & Angle Parse Logic ─────────────────────────────────────
    if args.title:
        title = args.title
    else:
        # Look for patterns like '0graus' or '45graus' or '90deg' in the first filename
        filename_stem = os.path.splitext(os.path.basename(args.files[0]))[0]
        angle_match = re.search(r"(\d+)\s*(graus|grau|deg|degree)", filename_stem, re.IGNORECASE)
        
        if angle_match:
            angle_val = angle_match.group(1)
            title = f"Intensity comparison between HH and HV polarisation for {angle_val}°"
        else:
            title = f"Intensity comparison between HH and HV polarisation"

    # ── Subtitle Parsing (Deduplicated) ──────────────────────────────────────
    subtitles = [meta_subtitle(d[2]) for d in datasets]
    if len(subtitles) == 2 and subtitles[0] == subtitles[1]:
        subtitle = subtitles[0]   # Conditions match perfectly -> show on one clean line
    else:
        subtitle = "\n".join(f"  {lbl}: {sub}"
                             for lbl, sub in zip(labels, subtitles))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.11)

    for (x, y, meta), label, color in zip(datasets, labels, COLORS):
        ax.plot(x, y, color=color, lw=1.2, label=label)
        print(f"  {label:40s}  {len(x)} pts  "
              f"x: [{x[0]:.1f}, {x[-1]:.1f}] cm-1")

    ax.set_xlabel("Raman shift (cm\u207b\u00b9)", fontsize=11)
    ax.set_ylabel("Intensity (Cnt)",              fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.35)

    if len(args.files) > 1:
        ax.legend(fontsize=10)

    if args.xmin is not None or args.xmax is not None:
        ax.set_xlim(args.xmin, args.xmax)

    fig.suptitle(f"{title}\n{subtitle}", fontsize=10, y=0.97)

    if args.save:
        stem = os.path.splitext(args.files[0])[0]
        out  = stem + ("_overlay.png" if len(args.files) > 1 else "_plot.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()