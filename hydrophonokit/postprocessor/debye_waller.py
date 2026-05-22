"""
=============================================================================
  HydroPhonoKit Postprocessor — Debye-Waller Factors

  Computes mean square displacement and B-factors from phonon eigenvectors.

  Scientific Foundation:
    The Debye-Waller factor describes the attenuation of X-ray or neutron
    scattering due to thermal vibrations. It is characterized by the
    mean square displacement (MSD) of each atom:

      <u²>_i = (ℏ/2M_i) × Σ_qj |e_i(qj)|² / ω(qj) × coth(ℏω(qj)/2k_B T)

    where:
      - M_i is the mass of atom i
      - e_i(qj) is the eigenvector component for atom i in mode (q,j)
      - ω(qj) is the phonon frequency
      - coth(x) = cosh(x)/sinh(x) is the hyperbolic cotangent

    The isotropic B-factor (temperature factor) used in crystallography is:
      B_i = 8π² × <u²>_i / 3

    At T = 0, only zero-point motion contributes:
      <u²>_i(T=0) = (ℏ/2M_i) × Σ_qj |e_i(qj)|² / ω(qj)

    Applications:
      - X-ray/neutron diffraction refinement
      - Understanding thermal stability
      - Identifying rattling atoms in cage compounds
      - Distinguishing static vs dynamic disorder

  References:
    [1] Willis & Pryor, Thermal Vibrations in Crystallography (1975)
    [2] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT
    [3] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
=============================================================================
"""
import os
import numpy as np
from typing import Dict, Optional

from ..physics import HBAR, K_BOLTZMANN


class DebyeWallerComputer:
    """Computes Debye-Waller factors from phonon eigenvectors.

    This class handles:
      1. Mean square displacement <u²> for each atom
      2. Isotropic B-factor computation
      3. Anisotropic displacement parameters (ADP) tensor
      4. Temperature dependence
    """

    def __init__(self, phonon, profile):
        """
        Args:
            phonon: Phonopy object with eigenvectors available
            profile: MaterialProfile from analyzer
        """
        self.phonon = phonon
        self.profile = profile

    def compute(self, temperature: float = 300,
                q_mesh: Optional[tuple] = None) -> Dict:
        """Compute Debye-Waller factors.

        Args:
            temperature: Temperature in Kelvin
            q_mesh: Q-point mesh for DOS computation (default: [15, 15, 15])

        Returns:
            dict: {
                'temperature': float (K),
                'mean_square_displacement': array (n_atoms, 3) in Å²,
                'isotropic_u2': array (n_atoms) in Å²,
                'B_factor': array (n_atoms) in Å²,
                'adp_tensor': array (n_atoms, 3, 3) in Å²,
                'zero_point_u2': array (n_atoms) in Å²,  # T=0 contribution
            }
        """
        print(f"\n[Scientific] Computing Debye-Waller Factors at {temperature}K...")

        if q_mesh is None:
            q_mesh = [15, 15, 15]

        n_atoms = len(self.phonon.unitcell.numbers)
        masses = np.array([self.phonon.unitcell.masses[i] for i in range(n_atoms)])

        # Compute phonon DOS with eigenvectors
        self.phonon.run_mesh(q_mesh, is_mesh_symmetry=True, with_eigenvectors=True)

        frequencies = self.phonon.mesh_frequencies  # (n_q, n_bands)
        eigenvectors = self.phonon.eigenvectors  # (n_q, n_bands, n_atoms, 3)
        weights = self.phonon.mesh_weights  # (n_q,)

        # Initialize accumulators
        msd = np.zeros((n_atoms, 3))  # <u²> per atom per direction
        adp = np.zeros((n_atoms, 3, 3))  # Anisotropic displacement parameters

        # Physical constants
        # HBAR in eV·s, K_BOLTZMANN in eV/K
        # Need to convert to consistent units (THz, Å)
        # ω in THz → ℏω in eV: ℏ (eV·s) × ω (THz × 1e12) = HBAR × ω × 1e12
        THZ_TO_EVS = 1e12  # THz → Hz

        for iq in range(frequencies.shape[0]):
            for ib in range(frequencies.shape[1]):
                freq = frequencies[iq, ib]

                # Skip imaginary/near-zero frequencies
                if freq < 0.01:
                    continue

                # Bose-Einstein factor: coth(ℏω/2kT) = 1 + 2/(e^(ℏω/kT) - 1)
                hbar_omega_ev = HBAR * freq * THZ_TO_EVS
                kT_ev = K_BOLTZMANN * temperature

                if kT_ev > 0:
                    x = hbar_omega_ev / kT_ev
                    if x > 100:
                        coth_x = 1.0  # T → 0 limit
                    else:
                        coth_x = np.cosh(x) / np.sinh(x)
                else:
                    coth_x = 1.0  # T = 0

                # Eigenvector contribution: |e_i(qj)|²
                # eigenvectors[iq, ib, i, α] is complex
                e_squared = np.abs(eigenvectors[iq, ib]) ** 2  # (n_atoms, 3)

                # Weight factor: ℏ/(2Mω) × coth(ℏω/2kT)
                # Mass in amu → kg: M_amu × 1.66054e-27
                # ℏ in eV·s, ω in THz → 1e12 Hz
                # Result in m² → convert to Å² (× 1e20)
                AMU_TO_KG = 1.66054e-27
                M_TO_A2 = 1e20  # m² → Å²

                weight = (HBAR / (2 * freq * THZ_TO_EVS)) * coth_x  # eV·s²

                # Convert to Å²: weight × (1/M) × conversion
                for i in range(n_atoms):
                    m_kg = masses[i] * AMU_TO_KG
                    factor = weight / m_kg * M_TO_A2

                    msd[i] += e_squared[i] * factor * weights[iq]
                    adp[i] += np.outer(e_squared[i], e_squared[i]) * factor * weights[iq]

        # Isotropic MSD (average over 3 directions)
        iso_u2 = np.mean(msd, axis=1)

        # B-factor: B = 8π² × <u²>/3
        B_factor = 8 * np.pi ** 2 * iso_u2 / 3

        # Zero-point contribution (T = 0, coth = 1)
        # Simplified: recompute with coth = 1 only
        zpe_u2 = self._compute_zero_point_u2(masses, frequencies, eigenvectors, weights)

        print(f"  --> Computed Debye-Waller factors for {n_atoms} atoms.")
        print(f"  --> Average <u²>: {np.mean(iso_u2):.4f} Å²")
        print(f"  --> Average B-factor: {np.mean(B_factor):.2f} Å²")

        return {
            'temperature': temperature,
            'mean_square_displacement': msd,
            'isotropic_u2': iso_u2,
            'B_factor': B_factor,
            'adp_tensor': adp,
            'zero_point_u2': zpe_u2,
        }

    def _compute_zero_point_u2(self, masses, frequencies, eigenvectors, weights):
        """Compute zero-point mean square displacement (T = 0).

        At T = 0, coth(ℏω/2kT) → 1, so only zero-point motion contributes.
        """
        n_atoms = len(masses)
        zpe_u2 = np.zeros(n_atoms)

        THZ_TO_EVS = 1e12
        AMU_TO_KG = 1.66054e-27
        M_TO_A2 = 1e20

        for iq in range(frequencies.shape[0]):
            for ib in range(frequencies.shape[1]):
                freq = frequencies[iq, ib]
                if freq < 0.01:
                    continue

                e_squared = np.abs(eigenvectors[iq, ib]) ** 2
                weight = HBAR / (2 * freq * THZ_TO_EVS)

                for i in range(n_atoms):
                    m_kg = masses[i] * AMU_TO_KG
                    factor = weight / m_kg * M_TO_A2
                    zpe_u2[i] += np.sum(e_squared[i]) * factor * weights[iq]

        return zpe_u2
