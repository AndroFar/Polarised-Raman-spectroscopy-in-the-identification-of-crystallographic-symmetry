"""
Raman Intensity Calculator for Fd-3m (Oh point group)
======================================================
Computes Raman scattering intensities for A1g, Eg, and T2g modes
under arbitrary crystal orientation and scattering geometry.

Key difference from F-43m (Td):
    Fd-3m has an inversion center, so the mutual exclusion rule applies:
    IR-active and Raman-active modes are mutually exclusive.
    Raman-active modes are gerade (even): A1g, Eg, T2g.
    The canonical example is silicon: one T2g mode at ~520 cm-1.

Physics:
    I proportional to |e_s . R . e_i|^2

where:
    e_i  = incident polarization vector (lab frame)
    e_s  = scattered polarization vector (lab frame)
    R    = Raman tensor (crystal frame)

Polarization vectors are rotated into the crystal frame via:
    e_crystal = M^T . e_lab

where M is the ZYZ rotation matrix

                          [cos(Θ)cos(φ)cos(ψ) - sin(φ)sin(ψ)   -cos(Θ)sin(φ)cos(ψ) - cos(φ)sin(ψ)   sin(Θ)cos(ψ)]
    r = Rz(ψ)Ry(Θ)Rz(φ) = [cos(Θ)cos(φ)sin(ψ) + sin(φ)cos(ψ)   -cos(Θ)sin(φ)sin(ψ) + cos(φ)cos(ψ)   sin(Θ)sin(ψ)]
                          [        - sin(Θ)cos(φ)                         sin(Θ)sin(φ)                  cos(Θ)  ]

where (φ, Θ, ψ) are the euler angles, representing...
"""

import numpy as np

# ---------------------------------------------------------------------------
# Rotation matrices (ZYZ convention)
# ---------------------------------------------------------------------------

def Rz(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=float)

def Ry(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c,  0,  s],
                     [ 0,  1,  0],
                     [-s,  0,  c]], dtype=float)

def euler_rotation(phi, theta, psi):
    """ZYZ Euler rotation: M = Rz(psi) . Ry(theta) . Rz(phi)"""
    return Rz(psi) @ Ry(theta) @ Rz(phi)

# ---------------------------------------------------------------------------
# Raman tensors for Fd-3m (Oh point group)
# ---------------------------------------------------------------------------

def get_tensors(a, b, d):
    """
    Raman tensors for Oh point group.

    Parameters
    ----------
    a : float  - A1g tensor element
    b : float  - Eg tensor element
    d : float  - T2g tensor element  (the relevant one for silicon)

    Tensor structures are identical in form to the Td case,
    but carry gerade labels due to the presence of inversion symmetry.

    For silicon at the zone centre only T2g is present (a=b=0, d!=0).
    """
    return {
        "A1g": [
            np.array([[a, 0, 0],
                      [0, a, 0],
                      [0, 0, a]], dtype=float)
        ],
        "Eg": [
            # Eg_theta: basis function 2z^2 - x^2 - y^2
            np.array([[  b,           0,    0],
                      [  0,           b,    0],
                      [  0,           0, -2*b]], dtype=float),
            # Eg_epsilon: basis function x^2 - y^2
            np.array([[-np.sqrt(3)*b,           0, 0],
                      [           0, np.sqrt(3)*b, 0],
                      [           0,           0, 0]], dtype=float),
        ],
        "T2g": [
            # T2g(x): non-zero in yz/zy
            np.array([[0, 0, 0],
                      [0, 0, d],
                      [0, d, 0]], dtype=float),
            # T2g(y): non-zero in xz/zx
            np.array([[0, 0, d],
                      [0, 0, 0],
                      [d, 0, 0]], dtype=float),
            # T2g(z): non-zero in xy/yx
            np.array([[0, d, 0],
                      [d, 0, 0],
                      [0, 0, 0]], dtype=float),
        ],
    }

# ---------------------------------------------------------------------------
# Porto notation -> polarization vectors (lab frame)
# ---------------------------------------------------------------------------

x = np.array([1, 0, 0], dtype=float)
y = np.array([0, 1, 0], dtype=float)
z = np.array([0, 0, 1], dtype=float)

# Tensors in Oh are symmetric, so -a(bc)a = -a(cb)a.
# Redundant crossed geometries are excluded.

PORTO_GEOMETRIES = {
    # backscattering along x
    "-x(yy)x": (y, y),
    "-x(yz)x": (y, z),
    "-x(zz)x": (z, z),
    # backscattering along y
    "-y(xx)y": (x, x),
    "-y(xz)y": (x, z),
    "-y(zz)y": (z, z),
    # backscattering along z
    "-z(xx)z": (x, x),
    "-z(xy)z": (x, y),
    "-z(yy)z": (y, y),
}

# ---------------------------------------------------------------------------
# Core intensity calculation
# ---------------------------------------------------------------------------

PARTNER_LABELS = {
    "A1g": ["A1g"],
    "Eg":  ["Eg_theta", "Eg_epsilon"],
    "T2g": ["T2g(x)", "T2g(y)", "T2g(z)"],
}

def raman_intensity(ei_lab, es_lab, M, tensors):
    ei_c = M.T @ ei_lab
    es_c = M.T @ es_lab
    results = {}
    for mode, tensor_list in tensors.items():
        partners = []
        total = 0.0
        for R in tensor_list:
            scalar = es_c @ R @ ei_c
            intensity = scalar ** 2
            total += intensity
            partners.append({"scalar": scalar, "intensity": intensity})
        results[mode] = {"intensity": total, "partners": partners}
    return results, ei_c, es_c

def print_results(porto, phi, theta, psi, a, b, d, results, ei_c, es_c):
    print()
    print("=" * 60)
    print("  Raman intensity - Fd-3m (Oh)")
    print("=" * 60)
    print(f"  Geometry              : {porto}")
    print(f"  Euler (phi,theta,psi) : ({phi}deg, {theta}deg, {psi}deg)")
    print(f"  Tensor params         : a={a}, b={b}, d={d}")
    print()
    print(f"  e_i (crystal frame) : [{ei_c[0]:+.4f}, {ei_c[1]:+.4f}, {ei_c[2]:+.4f}]")
    print(f"  e_s (crystal frame) : [{es_c[0]:+.4f}, {es_c[1]:+.4f}, {es_c[2]:+.4f}]")
    print()
    print(f"  {'Mode':<12} {'Intensity':>12}  {'Note'}")
    print(f"  {'-'*12} {'-'*12}  {'-'*20}")
    notes = {
        "A1g": "not present in Si",
        "Eg":  "not present in Si",
        "T2g": "Si ~520 cm-1",
    }
    for mode, data in results.items():
        print(f"  {mode:<12} {data['intensity']:>12.6f}  {notes[mode]}")
    print()
    print("  Partner breakdown:")
    for mode, data in results.items():
        for i, p in enumerate(data["partners"]):
            label = PARTNER_LABELS[mode][i]
            print(f"    {label:<14}  e_s.R.e_i = {p['scalar']:+.6f}   |...|^2 = {p['intensity']:.6f}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def prompt_float(prompt, min_val=None, max_val=None):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "exit":
            raise SystemExit
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  Please enter a value >= {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Please enter a value <= {max_val}.")
                continue
            return val
        except ValueError:
            print("  Invalid input - please enter a number.")

def prompt_choice(prompt, choices):
    print(prompt)
    for i, c in enumerate(choices, 1):
        print(f"  [{i}] {c}")
    while True:
        raw = input("  Enter number: ").strip()
        if raw.lower() == "exit":
            raise SystemExit
        try:
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            print(f"  Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("  Invalid input - please enter a number.")

# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  Raman Intensity Calculator - Fd-3m (Oh)")
    print("  Example material: Silicon (T2g at ~520 cm-1)")
    print("=" * 60)
    print()
    print("  For silicon: set a=0, b=0, d=<your T2g amplitude>")
    print("  Type 'exit' at any prompt to quit.")
    print()

    porto_options = list(PORTO_GEOMETRIES.keys())

    while True:
        try:
            print("-" * 60)
            porto = prompt_choice(
                "Select scattering geometry (Porto notation):",
                porto_options
            )

            print()
            print("Crystal orientation - ZYZ Euler angles")
            print("  (0, 0, 0) = crystal axes aligned with lab axes")
            phi   = prompt_float("  phi   [0-360 deg]: ", 0, 360)
            theta = prompt_float("  theta [0-180 deg]: ", 0, 180)
            psi   = prompt_float("  psi   [0-360 deg]: ", 0, 360)

            print()
            print("Raman tensor parameters")
            print("  (for silicon use a=0, b=0)")
            a = prompt_float("  a (A1g element) : ")
            b = prompt_float("  b (Eg element)  : ")
            d = prompt_float("  d (T2g element) : ")

            ei_lab, es_lab = PORTO_GEOMETRIES[porto]
            M       = euler_rotation(phi, theta, psi)
            tensors = get_tensors(a, b, d)
            results, ei_c, es_c = raman_intensity(ei_lab, es_lab, M, tensors)
            print_results(porto, phi, theta, psi, a, b, d, results, ei_c, es_c)

            print()
            again = input("  Run another calculation? [y/n]: ").strip().lower()
            if again != "y":
                print()
                print("  Goodbye.")
                break

        except SystemExit:
            print()
            print("  Goodbye.")
            break
        except (KeyboardInterrupt, EOFError):
            print()
            print("  Goodbye.")
            break

if __name__ == "__main__":
    main()