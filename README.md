# Polarised Raman Spectroscopy in the Identification of Crystallographic Symmetry

Analysis and simulation code developed for the final project of the BSc in Physics at the **University of Aveiro (UA)**, 2025/2026.

The project investigates how polarisation-resolved Raman scattering can be used to determine crystal symmetry, using crystalline silicon as the model system. This repository contains the Python tooling used to fit measured spectra, extract polarisation-dependent intensities, and compare them with the theoretical predictions from Raman tensor formalism.

---

## Table of Contents

- [Project Objective](#project-objective)
- [Background](#background)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Data Format](#data-format)
- [Usage](#usage)
- [Methods Overview](#methods-overview)
- [Known Limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Objective

The primary goal is to automate the tedious process of examining large quantities of Raman spectra. Rather than treating each spectrum individually, this pipeline is capable of processing. Although the pipeline was originally supposed to be tested on both silicon and gallium arsenide, time constraints restricted testing to just the silicon sample.

## Background

Raman scattering probes the vibrational modes (phonons) of a crystal. Whether a given mode is Raman-active, and how its intensity depends on the polarisation of the incident and scattered light, is dictated by the crystal's point-group symmetry through the **Raman tensor** of each mode.

For crystalline silicon (point group $O_h$), the first-order Raman spectrum is dominated by the triply degenerate **$T_{2g}$** optical phonon near $520\ \mathrm{cm}^{-1}$. In a **backscattering geometry**, described using **Porto notation**, the intensity of this mode varies in a characteristic way as the sample or polarisation is rotated, and differs between parallel (**HH**) and crossed (**HV**) polarisation configurations. Measuring these angular dependences and comparing them with the tensor-derived selection rules allows the symmetry of the scattering volume to be identified.

## Repository Structure

```
.
├── Simulations/        # Coupled mass-spring system and 1D diatomic lattice dispersion
│   ├── derive_CMSS.ipynb
│   ├── animate_CMSS.py
│   └── DR_Diatomic.ipynb
├── Result Analysis/    # Fitting, visualisation and polar-plot scripts for experimental spectra
│   ├── spectrum_visualiser.py
│   ├── raman_lorentzian_fit.py
│   ├── raman_polar_norm_HH.py
│   └── raman_polar_norm_HV.py
├── Extras/             # Raman intensity calculators for arbitrary orientation
│   ├── fd3m_Int_Calc.py
│   └── f43m_Int_Calc.py
├── LICENSE             # MIT License
└── README.md
```

| Directory | Purpose |
|---|---|
| `Simulations/` | Lattice-dynamics toy models used to build intuition for coupled phonon modes: a symbolic derivation and animation of a coupled mass-spring system, and the dispersion relation of a perfect 1D diatomic lattice. |
| `Result Analysis/` | Command-line tools for the experimental pipeline: spectrum inspection, interactive baseline correction, Lorentzian peak fitting over a batch of angles, and normalised polar plots compared against the theoretical HH and HV angular dependences. |
| `Extras/` | Interactive calculators of Raman selection rules and relative mode intensities for the two cubic space groups Fd-3m (e.g. silicon) and F-43m (e.g. zinc-blende compounds such as GaAs), for any scattering geometry and crystal orientation. These sit outside the main fitting pipeline. |

### Simulation scripts

| File | Description |
|---|---|
| `derive_CMSS.ipynb` | SymPy derivation of the equations of motion of a coupled mass-spring system (CMSS): two masses connected by three springs between fixed walls. Builds the Lagrangian from the kinetic and potential energies and solves the Euler-Lagrange equations for the accelerations of both masses. |
| `animate_CMSS.py` | Integrates the equations derived above with `scipy.integrate.solve_ivp` and produces a Matplotlib animation of the two masses and three springs, exported as `CMSS.mp4`. Masses, spring constants, initial displacements, duration and frame rate are set at the top of the script. |
| `DR_Diatomic.ipynb` | Dispersion relation $\omega(q)$ of a perfect 1D diatomic lattice with alternating spring constants and two different masses. Derives the dynamical matrix, solves its determinant for $\omega^2(q)$, and plots the acoustic and optical branches over the first Brillouin zone. |

### Analysis scripts

| Script | Description |
|---|---|
| `spectrum_visualiser.py` | Reads one or two LabRAM/Horiba `.txt` spectra and overlays them (for example HH against HV at the same angle). Parses the acquisition metadata (laser wavelength, grating, objective, acquisition time and accumulations) into the plot subtitle, supports `--xmin`/`--xmax` zooming and `--save` to PNG. |
| `raman_lorentzian_fit.py` | Batch Lorentzian fitting of the main peak. For each spectrum it opens an interactive baseline editor, subtracts a linear baseline, fits $L(x) = \dfrac{A\gamma^2}{(x-x_0)^2+\gamma^2}$, and saves a fit plot with residuals plus a results file. Extracts the rotation angle from each filename and writes `fit_summary.csv` (peak position, height, HWHM/FWHM, uncertainties, residual RMS), followed by an intensity-versus-angle plot. |
| `raman_polar_norm_HH.py` | Reads `fit_summary.csv`, computes the integrated area ($A\gamma\pi$) with propagated uncertainty, normalises peak height and area to $[0, 1]$, and plots them in polar form against the HH theory curve $\sin^2\!\big(2(\theta-\theta_0)\big)$. Also writes an angle-intensity table (`.txt` and `.csv`). |
| `raman_polar_norm_HV.py` | Same analysis for the crossed (HV) configuration, compared with $\cos^2\!\big(2(\theta-\theta_0)\big)$. |

### Extras scripts

| Script | Description |
|---|---|
| `fd3m_Int_Calc.py` | Raman intensity calculator for **Fd-3m** (point group $O_h$), the space group of silicon. Computes the intensity of the $A_{1g}$, $E_g$ and $T_{2g}$ modes from $I \propto \lvert \mathbf{e}_s \cdot R \cdot \mathbf{e}_i \rvert^2$. Because the crystal has an inversion centre, only the gerade modes are Raman-active; for silicon only the $T_{2g}$ mode near $520\ \mathrm{cm}^{-1}$ is present. |
| `f43m_Int_Calc.py` | The same calculation for **F-43m** (point group $T_d$, the zinc-blende structure), covering the $A_1$, $E$ and $T_2$ modes. |

Both calculators work as follows: the incident and scattered polarisation vectors for a chosen backscattering geometry in Porto notation (for example `-z(xy)z`) are rotated into the crystal frame using a ZYZ Euler rotation $M = R_z(\psi)\,R_y(\theta)\,R_z(\phi)$, then contracted with the Raman tensor of each irreducible representation. Output includes the intensity per mode and the contribution of each degenerate partner. Nine independent backscattering geometries are provided; redundant ones are omitted because the tensors are symmetric.

## Getting Started

### Prerequisites

- Python 3.9 or newer
- NumPy, SciPy, Matplotlib (and pandas, if used for data loading)
- SymPy and Jupyter, for the derivation and dispersion notebooks
- A GUI-capable Matplotlib backend (Tk or Qt), needed for the interactive baseline editor in `raman_lorentzian_fit.py`
- [FFmpeg](https://ffmpeg.org/), required by `animate_CMSS.py` to write the `.mp4` animation

### Installation

```bash
git clone https://github.com/AndroFar/polarised-raman-spectroscopy-in-the-identification-of-crystallographic-symmetry.git
cd polarised-raman-spectroscopy-in-the-identification-of-crystallographic-symmetry

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pandas sympy jupyter
```

## Data Format

Spectra were acquired on a **Horiba LabRAM** system and exported as `.txt` files. The exports use the **European decimal-comma** convention and begin with a block of `#`-prefixed `key = value` metadata lines (title, laser, grating, objective, acquisition time, accumulations, spectral resolution, date). The provided scripts handle both automatically.

After the header, each spectrum is two columns: Raman shift ($\mathrm{cm}^{-1}$) and intensity (counts). The stage rotation angle is read from the **filename**, which should contain a number followed by `graus`, `degrees` or `deg` (for example `HH_45graus.txt`), or a trailing number after a `-` or `_` separator.

The fitting script writes `fit_summary.csv`, which is the input to both polar-plot scripts.

## Usage

### Simulations

```bash
# Open the symbolic derivation and the dispersion-relation notebooks
jupyter notebook Simulations/derive_CMSS.ipynb
jupyter notebook Simulations/DR_Diatomic.ipynb

# Render the coupled mass-spring animation (writes CMSS.mp4 to the working directory)
python Simulations/animate_CMSS.py
```

The simulations are self-contained and need no experimental data. To explore different regimes, edit the masses (`m1`, `m2`), spring constants (`k1`, `k2`, `k3`) and initial conditions at the top of `animate_CMSS.py`, or the parameters at the top of the plotting cell in `DR_Diatomic.ipynb`.

### Experimental analysis

All scripts are run from the repository root and take their input on the command line.

```bash
# 0. (Optional) Inspect spectra, e.g. overlay HH and HV at the same angle
python "Result Analysis/spectrum_visualiser.py" HH_45graus.txt HV_45graus.txt \
    --labels HH HV --xmin 480 --xmax 560 --save

# 1. Fit the T2g peak for every .txt spectrum in a folder (one HH or HV series at a time).
#    An interactive baseline editor opens for each file; results go to <folder>/results.
python "Result Analysis/raman_lorentzian_fit.py" path/to/HH_spectra/

# 2. Generate polar plots from the resulting fit summary
python "Result Analysis/raman_polar_norm_HH.py" path/to/HH_spectra/results/fit_summary.csv
python "Result Analysis/raman_polar_norm_HV.py" path/to/HV_spectra/results/fit_summary.csv
```

Useful options for the fitting script:

| Option | Effect |
|---|---|
| `--output DIR` | Write results to `DIR` instead of `<input>/results`. |
| `--skip-baseline` | Define the baseline regions once on the first file and reuse them for all others. |
| `--col-x`, `--col-y` | Select the x and y columns (0-indexed, default 0 and 1). |

The polar scripts accept `--output DIR` and write their PNG figures and angle-intensity tables next to the input CSV by default.

### Intensity calculators

```bash
python Extras/fd3m_Int_Calc.py     # Fd-3m (O_h), e.g. silicon
python Extras/f43m_Int_Calc.py     # F-43m (T_d), e.g. GaAs
```

Both scripts are interactive and need only NumPy. You are prompted for:

1. a scattering geometry in Porto notation, chosen from a numbered list;
2. the crystal orientation as ZYZ Euler angles $\phi, \theta, \psi$ in degrees, where $(0, 0, 0)$ means crystal axes aligned with the lab axes;
3. the Raman tensor amplitudes `a`, `b` and `d` ($A$, $E$ and $T_2$ elements respectively).

For silicon, set `a = 0` and `b = 0` and use any non-zero `d`; since the result is a relative intensity, only the angular dependence matters. Type `exit` at any prompt to quit.

## Methods Overview

0. **Lattice-dynamics models.** A coupled mass-spring system (Lagrangian mechanics, solved symbolically and numerically) and a 1D diatomic chain (dynamical matrix, acoustic and optical branches) illustrate how coupled oscillators give rise to the phonon modes probed by Raman scattering.
1. **Baseline correction.** A linear baseline is fitted to two user-selected flat regions on either side of the peak, then subtracted. The script warns if the corrected flat regions do not sit near zero, which indicates the regions overlap the peak shoulders.
2. **Peak fitting.** The first-order silicon band is fitted with a Lorentzian at each angle, giving the peak position, height $A$ and HWHM $\gamma$ with uncertainties from the fit covariance.
3. **Intensity extraction.** Both the peak height and the integrated area ($A\gamma\pi$) are used as intensity measures, each normalised to its maximum.
4. **Angular analysis and comparison with theory.** Normalised intensities are plotted in polar form against the stage angle and compared with the Raman-tensor predictions for backscattering from a (001)-type face of a cubic crystal:

   $$I_{\mathrm{HH}} \propto \sin^2\!\big(2(\theta-\theta_0)\big), \qquad I_{\mathrm{HV}} \propto \cos^2\!\big(2(\theta-\theta_0)\big)$$

   where $\theta_0$ is the offset between the sample axis and the stage zero.
5. **General orientation.** The calculators in `Extras/` extend the same tensor formalism to arbitrary crystal orientations and scattering geometries, and can be used to predict the expected angular dependence for a sample cut or mounted differently.

## Known Limitations

- The angular offset $\theta_0$ between sample axes and the stage zero is a manually tuned constant (`OFFSET_DEG = 51.0` in each polar script) rather than a fitted parameter with uncertainty.
- Normalising to the empirical maximum introduces a small bias from noise.
- Area uncertainties do not currently include covariance cross-terms between fit parameters.
- The polar scripts skip rows flagged `bad_fit` in `fit_summary.csv`, but `raman_lorentzian_fit.py` does not currently write that column, so no fits are excluded automatically.
- The intensity calculators return relative intensities in arbitrary units set by the tensor amplitudes. Optical effects such as Fresnel transmission or polarisation mixing from high-aperture objectives are not modelled.

These are documented for transparency and would be natural starting points for future improvement.

## Acknowledgements

Project carried out at the University of Aveiro under the supervision of Dra. Maria do Rosário P. Correia and Dr. José Pedro S. Cardoso.

## License

Released under the [MIT License](LICENSE).
