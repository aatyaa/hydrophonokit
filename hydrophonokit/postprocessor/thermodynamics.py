"""
=============================================================================
  HydroPhonoKit Postprocessor — Thermodynamics Computer

  Computes vibrational thermodynamic properties in the harmonic approximation.

  Scientific Foundation:
    In the harmonic approximation, the phonon free energy is:
      F(T) = k_B T × Σ_qj ln[2 sinh(ℏω(qj)/2k_B T)]

    From F(T), other thermodynamic quantities follow:
      S(T) = -∂F/∂T    (entropy)
      C_v(T) = -T × ∂²F/∂T²  (heat capacity at constant volume)

    At T → 0:
      F(0) = ZPE = ½ Σ ℏω  (Zero-Point Energy)
      S(0) → 0  (Third Law of Thermodynamics)

    At T → ∞:
      C_v → 3N × k_B  (Dulong-Petit limit)

    Validation against these limits ensures computational correctness.

  References:
    [1] Born & Huang, Dynamical Theory of Crystal Lattices (1954)
    [2] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT thermo
    [3] Grimvall, Thermophysical Properties of Materials (1999)
=============================================================================
"""
import os
import numpy as np
from typing import Dict

from ..physics import (
    dulong_petit_limit,
    zero_point_energy,
)


class ThermodynamicsComputer:
    """Computes vibrational thermodynamic properties.

    This class handles:
      1. F(T), S(T), C_v(T) computation from phonopy
      2. Third Law validation (S(0) → 0)
      3. Dulong-Petit limit check (C_v → 3NR at high T)
      4. Data export with metadata
    """

    def __init__(self, profile):
        """
        Args:
            profile: MaterialProfile from analyzer
        """
        self.profile = profile

    def compute(self, phonon) -> Dict:
        """Compute thermodynamic properties.

        Args:
            phonon: Phonopy object with force constants computed

        Returns:
            dict: {
                'thermo_data': dict (phonopy thermal properties),
                'validations': dict (Third Law, Dulong-Petit),
                'zpe': float (kJ/mol),
                'at_300K': dict (F, S, Cv at 300K),
            }
        """
        print("\n[Phase 4] Thermodynamic Integrals (Harmonic Approximation)")

        # Compute thermodynamics on dense mesh
        mesh = self.profile.rec_dos_mesh
        print(f"  --> Computing thermodynamics on {mesh[0]}x{mesh[1]}x{mesh[2]} Q-point mesh...")

        phonon.run_mesh(mesh)
        phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)

        thermo_data = phonon.get_thermal_properties_dict()
        temps = thermo_data['temperatures']
        free_energy = thermo_data['free_energy']  # kJ/mol
        entropy = thermo_data['entropy']  # J/(mol·K)
        heat_capacity = thermo_data['heat_capacity']  # J/(mol·K)

        # Zero-Point Energy
        zpe = free_energy[0]
        print(f"  --> Zero-Point Energy (ZPE): {zpe:.3f} kJ/mol")

        # Validate Third Law: S(T=0) should be → 0
        third_law_ok = abs(entropy[0]) < 0.01
        if third_law_ok:
            print(f"  [OK] Third Law validated: S(T=0) = {entropy[0]:.4e} J/(mol*K)")
        else:
            print(f"  [!] WARNING: S(T=0) = {entropy[0]:.4f} J/(mol·K) -- should be ~0")

        # 300K values
        idx_300 = int(np.argmin(np.abs(temps - 300)))
        at_300k = {
            'F_kJ_mol': float(free_energy[idx_300]),
            'S_J_molK': float(entropy[idx_300]),
            'Cv_J_molK': float(heat_capacity[idx_300]),
        }
        print(f"  --> Thermodynamics @ 300 K:")
        print(f"      F   = {at_300k['F_kJ_mol']:.3f} kJ/mol")
        print(f"      S   = {at_300k['S_J_molK']:.3f} J/(mol·K)")
        print(f"      C_v = {at_300k['Cv_J_molK']:.3f} J/(mol·K)")

        # Validate Dulong-Petit Limit
        n_atoms = len(phonon.unitcell.numbers)
        dp_limit = dulong_petit_limit(n_atoms)
        cv_1000k = heat_capacity[-1]
        dp_error = abs(cv_1000k - dp_limit) / dp_limit * 100

        dp_ok = dp_error <= 5
        print(f"  --> Dulong-Petit Limit Check (3N*R):")
        print(f"      Theoretical limit : {dp_limit:.1f} J/(mol·K)  (N={n_atoms})")
        print(f"      Calculated C_v@1000K: {cv_1000k:.1f} J/(mol·K)")
        print(f"      Deviation: {dp_error:.1f}%")

        if not dp_ok:
            print(f"  [!] WARNING: C_v at 1000K deviates {dp_error:.1f}% from Dulong-Petit.")
            print(f"      Material may have optical modes not yet converged.")

        validations = {
            'third_law_ok': third_law_ok,
            'entropy_at_0K': float(entropy[0]),
            'dulong_petit_limit': float(dp_limit),
            'cv_at_1000K': float(cv_1000k),
            'dulong_petit_error_pct': float(dp_error),
            'dulong_petit_ok': dp_ok,
        }

        # Save raw data
        self._save_thermo_data(thermo_data, n_atoms, dp_limit, zpe)

        return {
            'thermo_data': thermo_data,
            'validations': validations,
            'zpe': float(zpe),
            'at_300K': at_300k,
        }

    def _save_thermo_data(self, thermo_data, n_atoms, dp_limit, zpe):
        """Save thermodynamic properties to text file with metadata."""
        temps = thermo_data['temperatures']
        fe = thermo_data['free_energy']
        ent = thermo_data['entropy']
        cv = thermo_data['heat_capacity']

        output_dir = os.path.dirname(
            getattr(thermo_data, '__dict__', {}).get('_output_dir', '.')
        ) or '.'

        raw_path = os.path.join(output_dir, 'thermodynamic_properties.dat')

        with open(raw_path, 'w') as f:
            f.write("# HydroPhonoKit v2.7 -- Thermodynamic Properties (Harmonic Approximation)\n")
            f.write("# Reference: Born & Huang, Dynamical Theory of Crystal Lattices (1954)\n")
            f.write("#\n")
            f.write(f"# Material: {self.profile.formula}\n")
            f.write(f"# Atoms per cell: {n_atoms}\n")
            f.write(f"# Dulong-Petit limit: {dp_limit:.2f} J/(mol·K)\n")
            f.write(f"# ZPE: {zpe:.3f} kJ/mol\n")
            f.write("#\n")
            f.write("# T(K)   F(kJ/mol)   S(J/mol.K)   Cv(J/mol.K)\n")
            for t, ffe, s, c in zip(temps, fe, ent, cv):
                f.write(f"{t:6.1f}   {ffe:10.4f}   {s:10.4f}   {c:10.4f}\n")

        print(f"  --> Saved Integrals to: {raw_path}")
