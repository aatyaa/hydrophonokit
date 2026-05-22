"""
=============================================================================
  HydroPhonoKit v2.4 -- Quasi-Harmonic Approximation (QHA) Engine

  Computes thermal expansion, temperature-dependent properties, and
  thermodynamic quantities beyond the harmonic approximation.

  Scientific Foundation:
    - Harmonic approx: F(T) = E_DFT + F_phonon at fixed V
    - QHA: F(V,T) = E_DFT(V) + F_phonon(V,T)
    - Equilibrium: V_eq(T) = argmin_V F(V,T)
    - From V(T): alpha(T) = (1/V)(dV/dT)_P
                 B(T) = V(d^2F/dV^2)_T
                 C_p(T) = C_v(T) + T*V*alpha^2*B

  References:
    [1] Wallace, Thermodynamics of Crystals (1972)
    [2] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT review
    [3] Togo et al., Phys. Rev. B 81, 104301 (2010) -- QHA in phonopy
    [4] Grimvall, Thermophysical Properties of Materials (1999)
    [5] Moruzzi et al., Phys. Rev. B 37, 790 (1988) -- fcc metals QHA
=============================================================================
"""
import os
import json
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple

from .physics import R_GAS, H_PLANCK, K_BOLTZMANN, N_AVOGADRO, THZ_TO_CM
from .eos import (
    fit_eos, fit_all_eos, best_eos_model, birch_murnaghan,
    vinet, murnaghan, EOS_MODELS, EV_A3_TO_GPA, GPA_TO_EV_A3,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.linewidth': 1.2, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# ============================================================================
# DATA CLASSES
# ============================================================================

class QHAResult:
    """Container for QHA computation results."""

    def __init__(self):
        # Input
        self.formula = ""
        self.eos_model = "birch_murnaghan"
        self.n_volumes = 0

        # EOS fit results
        self.E0 = 0.0  # eV
        self.V0 = 0.0  # Angstrom^3
        self.B0 = 0.0  # GPa
        self.B0_prime = 4.0

        # Temperature-dependent arrays
        self.temperatures = np.array([])  # K
        self.volumes = np.array([])       # Angstrom^3
        self.free_energies = np.array([]) # kJ/mol, F(V_eq, T)

        # Derived properties
        self.alpha = np.array([])         # 1/K, volumetric thermal expansion
        self.B_T = np.array([])           # GPa, isothermal bulk modulus
        self.C_p = np.array([])           # J/(mol*K)
        self.gruneisen = np.array([])     # dimensionless

        # Harmonic reference (T=0)
        self.ZPE = 0.0  # kJ/mol

        # Fit quality
        self.eos_r_squared = {}  # {model: r^2}

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Quasi-Harmonic Approximation: {self.formula}")
        lines.append("=" * 60)
        lines.append(f"  EOS Model: {self.eos_model}")
        lines.append(f"  Volumes computed: {self.n_volumes}")
        lines.append("")

        lines.append("  EOS Fit Results (T = 0 K):")
        lines.append(f"    E0       = {self.E0:10.4f} eV")
        lines.append(f"    V0       = {self.V0:10.4f} A^3")
        lines.append(f"    B0       = {self.B0:10.2f} GPa")
        lines.append(f"    B0'      = {self.B0_prime:10.2f}")
        if self.eos_r_squared:
            for model, r2 in self.eos_r_squared.items():
                marker = " <--" if model == self.eos_model else ""
                lines.append(f"    R^2 ({model:20s}) = {r2:10.6f}{marker}")
        lines.append("")

        lines.append(f"  ZPE = {self.ZPE:.3f} kJ/mol")
        lines.append("")

        # 300K values
        if len(self.temperatures) > 0:
            idx_300 = int(np.argmin(np.abs(self.temperatures - 300)))
            lines.append("  Properties @ 300 K:")
            lines.append(f"    V(300K)   = {self.volumes[idx_300]:10.4f} A^3")
            lines.append(f"    alpha     = {self.alpha[idx_300]:10.2e} K^-1")
            lines.append(f"    B_T       = {self.B_T[idx_300]:10.2f} GPa")
            lines.append(f"    C_p       = {self.C_p[idx_300]:10.2f} J/(mol*K)")
            lines.append(f"    gamma     = {self.gruneisen[idx_300]:10.3f}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'formula': self.formula,
            'eos_model': self.eos_model,
            'n_volumes': self.n_volumes,
            'E0_eV': self.E0,
            'V0_A3': self.V0,
            'B0_GPa': self.B0,
            'B0_prime': self.B0_prime,
            'eos_r_squared': self.eos_r_squared,
            'ZPE_kJ_mol': self.ZPE,
            'temperatures_K': self.temperatures.tolist(),
            'volumes_A3': self.volumes.tolist(),
            'free_energy_kJ_mol': self.free_energies.tolist(),
            'alpha_K_inv': self.alpha.tolist(),
            'bulk_modulus_GPa': self.B_T.tolist(),
            'Cp_J_molK': self.C_p.tolist(),
            'gruneisen': self.gruneisen.tolist(),
        }


# ============================================================================
# QHA WORKFLOW ENGINE
# ============================================================================

class QHAEngine:
    """Quasi-Harmonic Approximation workflow engine.

    Requires phonon calculations at multiple volumes (typically 5-9).
    For each volume V_i:
        1. E_DFT(V_i) from static DFT
        2. F_phonon(V_i, T) from phonopy thermal properties

    Then:
        F(V,T) = E_DFT(V) + F_phonon(V,T)
        V_eq(T) = argmin_V F(V,T)
        alpha(T) = (1/V)(dV/dT)
        B(T) = V(d^2F/dV^2)
        C_p = C_v + T*V*alpha^2*B
    """

    def __init__(self, volume_dirs, formula="", profile=None):
        """
        Args:
            volume_dirs: List of (volume, energy_DFT, free_energies_vs_T) tuples
                         OR list of directories containing phonon results
            formula: Chemical formula
            profile: MaterialProfile (optional, for metadata)
        """
        self.volume_dirs = volume_dirs
        self.formula = formula
        self.profile = profile
        self.result = QHAResult()
        self.result.formula = formula
        self.result.n_volumes = len(volume_dirs)

    def execute(self, t_min=0, t_max=1000, t_step=10,
                eos_model='birch_murnaghan') -> QHAResult:
        """Execute full QHA workflow.

        Args:
            t_min: Minimum temperature (K)
            t_max: Maximum temperature (K)
            t_step: Temperature step (K)
            eos_model: EOS model name for fitting

        Returns:
            QHAResult with all temperature-dependent properties
        """
        result = self.result

        # Step 1: Parse input data
        volumes, energies_DFT, free_energies = self._parse_input()
        result.n_volumes = len(volumes)

        # Step 2: Fit EOS at T=0
        self._fit_eos(volumes, energies_DFT)

        # Step 3: Build F(V,T) matrix
        temperatures = np.arange(t_min, t_max + t_step, t_step, dtype=float)
        F_matrix = self._build_free_energy_matrix(volumes, free_energies, temperatures)

        # Step 4: Find V_eq(T) by minimizing F(V,T) at each T
        V_eq, F_min = self._find_equilibrium_volumes(volumes, F_matrix, temperatures)

        # Step 5: Compute thermal expansion alpha(T)
        alpha = self._compute_thermal_expansion(V_eq, temperatures)

        # Step 6: Compute bulk modulus B(T)
        B_T = self._compute_bulk_modulus(volumes, free_energies, temperatures, V_eq)

        # Store intermediate results required for spline calculations in Cp and Gruneisen
        result.temperatures = temperatures
        result.volumes = V_eq
        result.free_energies = F_min
        result.alpha = alpha
        result.B_T = B_T

        # Step 7: Compute C_p(T) from C_v(T) + correction
        C_p = self._compute_heat_capacity(free_energies, temperatures, V_eq, alpha, B_T)

        # Step 8: Compute Gruneisen parameter
        gruneisen = self._compute_gruneisen_parameter(alpha, B_T, V_eq, free_energies)

        # Store final derived properties
        result.C_p = C_p
        result.gruneisen = gruneisen

        return result

    def _parse_input(self):
        """Parse input volume directories or data tuples.

        Returns:
            volumes: array of volumes (A^3)
            energies_DFT: array of DFT energies (eV)
            free_energies: list of arrays, free energy vs T for each volume
        """
        volumes = []
        energies_DFT = []
        free_energies = []

        for item in self.volume_dirs:
            if isinstance(item, tuple):
                # Direct data: (V, E_DFT, F_phonon(T))
                V, E_DFT, F_T = item
                volumes.append(V)
                energies_DFT.append(E_DFT)
                free_energies.append(np.asarray(F_T))
            elif isinstance(item, dict):
                # Dictionary format
                volumes.append(item['volume'])
                energies_DFT.append(item['energy_DFT'])
                free_energies.append(np.asarray(item['free_energies_T']))
            else:
                raise ValueError(f"Unknown input format: {type(item)}")

        # Sort by volume
        sort_idx = np.argsort(volumes)
        volumes = np.array(volumes)[sort_idx]
        energies_DFT = np.array(energies_DFT)[sort_idx]
        free_energies = [free_energies[i] for i in sort_idx]

        return volumes, energies_DFT, free_energies

    def _fit_eos(self, volumes, energies_DFT):
        """Fit all EOS models at T=0."""
        results_all = fit_all_eos(volumes, energies_DFT)
        result = self.result

        # Store all R^2
        for model, res in results_all.items():
            if 'error' not in res:
                result.eos_r_squared[model] = res['r_squared']

        # Select best model
        best = best_eos_model(results_all)
        if best:
            result.eos_model = best
            best_res = results_all[best]
            result.E0 = best_res['E0']
            result.V0 = best_res['V0']
            result.B0 = best_res['B0']
            result.B0_prime = best_res['B0_prime']

    def _build_free_energy_matrix(self, volumes, free_energies, temperatures):
        """Build F(V,T) matrix of shape (n_volumes, n_temperatures).

        F(V,T) = E_DFT(V) + F_phonon(V,T)

        Returns:
            F_matrix: shape (n_V, n_T) in kJ/mol
        """
        n_V = len(volumes)
        n_T = len(temperatures)
        F_matrix = np.zeros((n_V, n_T))

        for i, F_T in enumerate(free_energies):
            if len(F_T) == n_T:
                F_matrix[i] = F_T  # Already includes E_DFT
            else:
                warnings.warn(
                    f"Volume {i} has {len(F_T)} T-points, expected {n_T}. "
                    f"Interpolating..."
                )
                T_orig = np.linspace(0, len(F_T) - 1, len(F_T)) * 10
                T_new = temperatures
                F_matrix[i] = np.interp(T_new, T_orig, F_T)

        return F_matrix

    def _find_equilibrium_volumes(self, volumes, F_matrix, temperatures):
        """Find V_eq(T) by minimizing F(V,T) at each temperature.

        Uses spline interpolation for smooth V(T).

        Returns:
            V_eq: equilibrium volumes at each T (A^3)
            F_min: minimum free energy at each T (kJ/mol)
        """
        from scipy.interpolate import UnivariateSpline

        n_T = len(temperatures)
        V_eq = np.zeros(n_T)
        F_min = np.zeros(n_T)

        for j in range(n_T):
            F_at_T = F_matrix[:, j]

            # Find minimum by spline fitting
            # Need at least 4 points for cubic spline
            if len(volumes) >= 4:
                try:
                    spline = UnivariateSpline(volumes, F_at_T, k=3, s=0)
                    # Minimize: find where dF/dV = 0
                    dspline = spline.derivative()
                    roots = dspline.roots()
                    if len(roots) > 0:
                        # Pick root within volume range
                        valid_roots = roots[(roots > volumes.min()) & (roots < volumes.max())]
                        if len(valid_roots) > 0:
                            V_eq[j] = valid_roots[np.argmin(np.abs(valid_roots - volumes[np.argmin(F_at_T)]))]
                        else:
                            V_eq[j] = volumes[np.argmin(F_at_T)]
                    else:
                        V_eq[j] = volumes[np.argmin(F_at_T)]
                    F_min[j] = spline(V_eq[j])
                except Exception:
                    V_eq[j] = volumes[np.argmin(F_at_T)]
                    F_min[j] = F_at_T.min()
            else:
                V_eq[j] = volumes[np.argmin(F_at_T)]
                F_min[j] = F_at_T.min()

        # Smooth V(T) with spline
        if len(temperatures) >= 4:
            try:
                V_spline = UnivariateSpline(temperatures, V_eq, k=3, s=0.1)
                V_eq = V_spline(temperatures)
            except Exception:
                pass

        return V_eq, F_min

    def _compute_thermal_expansion(self, V_eq, temperatures):
        """Compute volumetric thermal expansion coefficient.

        alpha(T) = (1/V)(dV/dT)_P

        Returns:
            alpha: 1/K
        """
        from scipy.interpolate import UnivariateSpline

        # Smooth V(T)
        V_spline = UnivariateSpline(temperatures, V_eq, k=3, s=0.1)
        dV_dT = V_spline.derivative()(temperatures)

        alpha = dV_dT / V_eq
        # Ensure non-negative
        alpha = np.maximum(alpha, 0.0)

        return alpha

    def _compute_bulk_modulus(self, volumes, free_energies, temperatures, V_eq):
        """Compute isothermal bulk modulus vs temperature.

        B(T) = V * (d^2F/dV^2)_T

        Returns:
            B_T: GPa
        """
        from scipy.interpolate import UnivariateSpline

        n_T = len(temperatures)
        B_T = np.zeros(n_T)

        for j in range(n_T):
            # Get F(V) at this temperature
            F_at_T = np.array([fe[j] if j < len(fe) else fe[-1]
                              for fe in free_energies])

            if len(volumes) >= 4:
                try:
                    spline = UnivariateSpline(volumes, F_at_T, k=3, s=0)
                    d2F_dV2 = spline.derivative(n=2)(V_eq[j])
                    # Convert: F in kJ/mol, V in A^3
                    # B = V * d2F/dV2
                    # 1 kJ/mol/A^3 = 1e3 / (N_A * 1e-30) Pa = 1e3 * 1e30 / N_A Pa
                    # = 1e33 / 6.022e23 Pa = 1.6605e9 Pa = 1.6605 GPa
                    B_T[j] = V_eq[j] * d2F_dV2 * 1.6605
                except Exception:
                    B_T[j] = self.result.B0
            else:
                B_T[j] = self.result.B0

        return B_T

    def _compute_heat_capacity(self, free_energies, temperatures, V_eq, alpha, B_T):
        """Compute C_p from C_v + TV*alpha^2*B correction.

        C_p = C_v + T * V * alpha^2 * B

        Where C_v is approximated from phonon DOS.

        Returns:
            C_p: J/(mol*K)
        """
        # Approximate C_v from free energy second derivative
        # C_v = -T * d^2F/dT^2
        from scipy.interpolate import UnivariateSpline

        if len(temperatures) >= 4:
            F_spline = UnivariateSpline(temperatures, self.result.free_energies, k=3, s=1)
            d2F_dT2 = F_spline.derivative(n=2)(temperatures)
            C_v = -temperatures * d2F_dT2  # kJ/(mol*K)
            C_v = np.maximum(C_v, 0.0)
            C_v *= 1000  # kJ -> J
        else:
            C_v = np.zeros_like(temperatures)

        # C_p correction
        V_m3 = V_eq * 1e-30 * N_AVOGADRO  # m^3/mol
        C_p = C_v + temperatures * V_m3 * alpha**2 * B_T * 1e9  # J/(mol*K)

        return C_p

    def _compute_gruneisen_parameter(self, alpha, B_T, V_eq, free_energies):
        """Compute macroscopic Gruneisen parameter.

        gamma = alpha * B_T * V / C_v

        Returns:
            gruneisen: dimensionless
        """
        V_m3 = V_eq * 1e-30 * N_AVOGADRO  # m^3/mol

        # C_v from free energy
        from scipy.interpolate import UnivariateSpline
        if len(self.result.temperatures) >= 4:
            F_spline = UnivariateSpline(self.result.temperatures,
                                        self.result.free_energies, k=3, s=1)
            d2F_dT2 = F_spline.derivative(n=2)(self.result.temperatures)
            C_v = -self.result.temperatures * d2F_dT2 * 1000  # J/(mol*K)
            C_v = np.maximum(C_v, 1.0)  # Avoid division by zero
        else:
            C_v = np.ones_like(alpha) * R_GAS * 3

        gamma = alpha * B_T * 1e9 * V_m3 / C_v
        gamma = np.maximum(gamma, 0.0)

        return gamma

    def plot_results(self, output_dir):
        """Generate publication-quality QHA plots."""
        os.makedirs(output_dir, exist_ok=True)
        r = self.result

        if len(r.temperatures) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # V(T)
        ax = axes[0, 0]
        ax.plot(r.temperatures, r.volumes, 'b-', lw=2)
        ax.set_xlabel('T (K)')
        ax.set_ylabel('V (A$^3$)')
        ax.set_title('Equilibrium Volume vs Temperature')
        ax.grid(alpha=0.2)

        # alpha(T)
        ax = axes[0, 1]
        ax.plot(r.temperatures, r.alpha * 1e6, 'r-', lw=2)
        ax.set_xlabel('T (K)')
        ax.set_ylabel(r'$\alpha$ ($10^{-6}$ K$^{-1}$)')
        ax.set_title('Thermal Expansion Coefficient')
        ax.grid(alpha=0.2)

        # B(T)
        ax = axes[1, 0]
        ax.plot(r.temperatures, r.B_T, 'g-', lw=2)
        ax.set_xlabel('T (K)')
        ax.set_ylabel('B (GPa)')
        ax.set_title('Bulk Modulus vs Temperature')
        ax.grid(alpha=0.2)

        # C_p(T)
        ax = axes[1, 1]
        ax.plot(r.temperatures, r.C_p, 'm-', lw=2, label=r'$C_p$')
        ax.set_xlabel('T (K)')
        ax.set_ylabel('C (J/mol$\cdot$K)')
        ax.set_title('Heat Capacity')
        ax.legend()
        ax.grid(alpha=0.2)

        fig.suptitle(f'QHA Results: {r.formula}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, 'qha_results.png')
        plt.savefig(path)
        plt.close()
        print(f"  [QHA] Saved plot: {path}")

        # Save data
        data_path = os.path.join(output_dir, 'qha_data.json')
        with open(data_path, 'w') as f:
            json.dump(r.to_dict(), f, indent=2)
        print(f"  [QHA] Saved data: {data_path}")
