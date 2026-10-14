"""
Raman Intensity Calculator for F-43m (Td point group)

Computes Raman scattering intensities for A1, E, and T2 modes
under arbitrary crystal orientation and scattering geometry.

Physics:
    I ∝ |ê_s · R · ê_i|²

where:
    ê_i  = incident polarization vector (lab frame)
    ê_s  = scattered polarization vector (lab frame)
    R    = Raman tensor (crystal frame)

When the crystal is rotated relative to the lab, the polarization
vectors are transformed into the crystal frame via:
    ê_crystal = M^T · ê_lab

M is the ZYZ convention rotation matrix

                          [cos(Θ)cos(φ)cos(ψ) - sin(φ)sin(ψ)   -cos(Θ)sin(φ)cos(ψ) - cos(φ)sin(ψ)   sin(Θ)cos(ψ)]
    r = Rz(ψ)Ry(Θ)Rz(φ) = [cos(Θ)cos(φ)sin(ψ) + sin(φ)cos(ψ)   -cos(Θ)sin(φ)sin(ψ) + cos(φ)cos(ψ)   sin(Θ)sin(ψ)]
                          [        - sin(Θ)cos(φ)                         sin(Θ)sin(φ)                  cos(Θ)  ]

where (φ, Θ, ψ) are the euler angles, representing...
"""

import numpy as np

# ---------------------------------------------------------------------------
# Rotation matrices
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
    return Rz(psi) @ Ry(theta) @ Rz(phi)

# ---------------------------------------------------------------------------
# Raman tensors for F-43m (Td)
# ---------------------------------------------------------------------------

def get_tensors(a, b, d):
    return {
        "A1": [np.array([[a,0,0],[0,a,0],[0,0,a]], dtype=float)],
        "E": [
            np.array([[b,0,0],[0,b,0],[0,0,-2*b]], dtype=float),
            np.array([[-np.sqrt(3)*b,0,0],[0,np.sqrt(3)*b,0],[0,0,0]], dtype=float),
        ],
        "T2": [
            np.array([[0,0,0],[0,0,d],[0,d,0]], dtype=float),
            np.array([[0,0,d],[0,0,0],[d,0,0]], dtype=float),
            np.array([[0,d,0],[d,0,0],[0,0,0]], dtype=float),
        ],
    }

# ---------------------------------------------------------------------------
# Porto notation -> polarization vectors (lab frame)
# ---------------------------------------------------------------------------

x = np.array([1,0,0], dtype=float)
y = np.array([0,1,0], dtype=float)
z = np.array([0,0,1], dtype=float)

# The tensors belonging to F-43m are all symmetric, so -a(bc)a = -a(cb)a.
# Redundant crossed geometries (-x(zy)x, -y(zx)y, -z(yx)z) are excluded.

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
    "A1": ["A1"],
    "E":  ["E_theta", "E_epsilon"],
    "T2": ["T2(x)", "T2(y)", "T2(z)"],
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
    print("  Raman intensity - F-43m (Td)")
    print("=" * 60)
    print(f"  Geometry              : {porto}")
    print(f"  Euler (phi,theta,psi) : ({phi}deg, {theta}deg, {psi}deg)")
    print(f"  Tensor params         : a={a}, b={b}, d={d}")
    print()
    print(f"  e_i (crystal frame) : [{ei_c[0]:+.4f}, {ei_c[1]:+.4f}, {ei_c[2]:+.4f}]")
    print(f"  e_s (crystal frame) : [{es_c[0]:+.4f}, {es_c[1]:+.4f}, {es_c[2]:+.4f}]")
    print()
    print(f"  {'Mode':<10} {'Intensity':>12}")
    print(f"  {'-'*10} {'-'*12}")
    for mode, data in results.items():
        print(f"  {mode:<10} {data['intensity']:>12.6f}")
    print()
    print("  Partner breakdown:")
    for mode, data in results.items():
        for i, p in enumerate(data["partners"]):
            label = PARTNER_LABELS[mode][i]
            print(f"    {label:<12}  e_s.R.e_i = {p['scalar']:+.6f}   |...|^2 = {p['intensity']:.6f}")
    print("=" * 60)

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

def main():
    print()
    print("=" * 60)
    print("  Raman Intensity Calculator - F-43m (Td)")
    print("=" * 60)
    print()
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
            a = prompt_float("  a (A1 element) : ")
            b = prompt_float("  b (E element)  : ")
            d = prompt_float("  d (T2 element) : ")

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
