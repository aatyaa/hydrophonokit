"""
=============================================================================
  HydroPhonoKit Postprocessor — Mode-Resolved Thermodynamics

  Computes element-projected thermodynamic properties.

  Scientific Foundation:
    The total phonon thermodynamic properties can be decomposed into
    contributions from individual atoms or elements:

      F(T) = k_B T × Σ_qj ln[2 sinh(ℏω(qj)/2k_B T)]
      F(T) = Σ_i F_i(T)

    where F_i(T) is the contribution from atom i, computed by projecting
    the phonon density of states onto atom i:

      g_i(ω) = Σ_qj |e_i(qj)|² δ(ω - ω(qj))

    Then:
      F_i(T) = k_B T × ∫ dω g_i(ω) ln[2 sinh(ℏω/2k_B T)]
      S_i(T) = -∂F_i/∂T
      C_v,i(T) = -T × ∂²F_i/∂T²

    This decomposition reveals which atoms dominate the entropy and
    heat capacity, essential for understanding:
      - Hydrogen's contribution to entropy in hydrides
      - Heavy vs light atom contributions
      - Design of materials with targeted thermal properties

  References:
    [1] Grimvall, Thermophysical Properties of Materials (1999)
    [2] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT
    [3] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy pDOS
=============================================================================
"""
import os
import numpy as np
from typing import Dict, Optional

from ..physics import (
    H_PLANCK, K_BOLTZMANN, HBAR, N_AVOGADRO, R_GAS,
    helmholtz_free_energy, phonon_entropy, heat_capacity_cv,
)


class ModeResolvedThermo:
    """Computes element-projected thermodynamic properties.

    This class handles:
      1. Per-element F(T), S(T), C_v(T) decomposition
      2. Per-mode thermodynamic contributions
      3. Hydrogen-specific entropy analysis
    """

    def __init__(self, phonon, profile):
        """
        Args:
            phonon: Phonopy object with eigenvectors available
            profile: MaterialProfile from analyzer
        """
        self.phonon = phonon
        self.profile = profile

    def compute(self, temperatures: Optional[np.ndarray] = None,
                q_mesh: Optional[tuple] = None) -> Dict:
        """Compute mode-resolved thermodynamics.

        Args:
            temperatures: Temperature array (default: 0-1000K, step 10)
            q_mesh: Q-point mesh for DOS computation (default: [15, 15, 15])

        Returns:
            dict: {
                'total': {'F': array, 'S': array, 'Cv': array},
                'by_element': {element: {'F': array, 'S': array, 'Cv': array}},
                'hydrogen_entropy_contribution': float (at 300K, J/(mol·K)),
                'element_fractions': {element: float at 300K},
            }
        """
        print("\n[Scientific] Computing Mode-Resolved Thermodynamics...")

        if temperatures is None:
            temperatures = np.arange(0, 1001, 10, dtype=float)
        if q_mesh is None:
            q_mesh = [15, 15, 15]

        n_atoms = len(self.phonon.unitcell.numbers)
        masses = self.phonon.unitcell.masses

        # Compute pDOS with eigenvectors
        self.phonon.run_mesh(q_mesh, is_mesh_symmetry=True, with_eigenvectors=True)
        self.phonon.run_projected_dos(freq_pitch=0.1, use_tetrahedron_method=False, sigma=0.2)

        pdos = self.phonon.projected_dos
        freq = pdos.frequency_points  # THz
        pdos_arr = pdos.projected_dos  # (n_atoms, n_freq)

        # Map atoms to elements
        from ..analyzer import Z_TO_SYMBOL
        sym_to_idx = {}
        for i, z in enumerate(self.phonon.unitcell.numbers):
            sym = Z_TO_SYMBOL.get(z, f"Z{z}")
            if sym not in sym_to_idx:
                sym_to_idx[sym] = []
            sym_to_idx[sym].append(i)

        # Total thermodynamics
        total_freq = pdos_arr.sum(axis=0)  # Total DOS
        total_F, total_S, total_Cv = self._compute_thermo_from_dos(total_freq, freq, temperatures)

        results = {
            'total': {
                'F': total_F,
                'S': total_S,
                'Cv': total_Cv,
            },
            'by_element': {},
            'hydrogen_entropy_contribution': 0.0,
            'element_fractions': {},
        }

        # Per-element thermodynamics
        for element, indices in sym_to_idx.items():
            # Sum partial DOS for this element
            elem_dos = pdos_arr[indices].sum(axis=0)

            F, S, Cv = self._compute_thermo_from_dos(elem_dos, freq, temperatures)
            results['by_element'][element] = {
                'F': F,
                'S': S,
                'Cv': Cv,
            }

            # Fraction of total entropy at 300K
            idx_300 = int(np.argmin(np.abs(temperatures - 300)))
            if total_S[idx_300] > 0:
                frac = S[idx_300] / total_S[idx_300] * 100
            else:
                frac = 0.0
            results['element_fractions'][element] = frac

        # Hydrogen-specific analysis
        if 'H' in sym_to_idx:
            h_S = results['by_element']['H']['S']
            idx_300 = int(np.argmin(np.abs(temperatures - 300)))
            results['hydrogen_entropy_contribution'] = float(h_S[idx_300])

        print(f"  --> Computed thermodynamics for {len(sym_to_idx)} elements.")
        for elem, frac in results['element_fractions'].items():
            print(f"      {elem}: {frac:.1f}% of total entropy at 300K")

        return results

    def _compute_thermo_from_dos(self, dos, freq, temperatures):
        """Compute F(T), S(T), C_v(T) from a density of states.

        Uses the harmonic approximation formulas:
          F(T) = k_B T × Σ ln[2 sinh(ℏω/2k_B T)] × g(ω) dω
          S(T) = -∂F/∂T
          C_v(T) = -T × ∂²F/∂T²

        For efficiency, we use phonopy's formulas applied to the DOS.

        Args:
            dos: Density of states array (n_freq)
            freq: Frequency array (THz)
            temperatures: Temperature array (K)

        Returns:
            (F, S, Cv) arrays, each (n_T,)
        """
        n_T = len(temperatures)
        F = np.zeros(n_T)
        S = np.zeros(n_T)
        Cv = np.zeros(n_T)

        # Integration step
        df = np.mean(np.diff(freq)) if len(freq) > 1 else 1.0

        for k, T in enumerate(temperatures):
            if T < 0.01:
                # T = 0: only zero-point energy
                # F = ½ × Σ ℏω × g(ω)
                F[k] = 0.5 * np.sum(HBAR * freq * 1e12 * dos * df) * N_AVOGADRO / 1000  # kJ/mol
                continue

            x = HBAR * freq * 1e12 / (K_BOLTZMANN * T)  # dimensionless

            # Free energy: F = kT × Σ ln[2 sinh(x/2)] × g(ω) dω
            # Handle x → 0 (low frequency) carefully
            safe_x = np.where(x < 1e-10, 1e-10, x)
            ln_sinh = np.log(2 * np.sinh(safe_x / 2))
            F[k] = K_BOLTZMANN * T * np.sum(ln_sinh * dos * df) * N_AVOGADRO / 1000  # kJ/mol

            # Entropy: S = -∂F/∂T = k_B × Σ [x/(2) coth(x/2) - ln(2 sinh(x/2))] × g(ω) dω
            coth_half = np.cosh(safe_x / 2) / np.sinh(safe_x / 2)
            entropy_integrand = (safe_x / 2) * coth_half - ln_sinh
            S[k] = K_BOLTZMANN * np.sum(entropy_integrand * dos * df) * N_AVOGADRO  # J/(mol·K)

            # Heat capacity: C_v = k_B × Σ (x/2)² / sinh²(x/2) × g(ω) dω
            sinh_half = np.sinh(safe_x / 2)
            cv_integrand = (safe_x / 2) ** 2 / sinh_half ** 2
            Cv[k] = K_BOLTZMANN * np.sum(cv_integrand * dos * df) * N_AVOGADRO  # J/(mol·K)

        return F, S, Cv
