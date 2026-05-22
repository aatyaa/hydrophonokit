"""
=============================================================================
  HydroPhonoKit v2.6 -- Hydrogen Storage Thermodynamics Engine

  Computes complete dehydrogenation thermodynamics from DFT + phonon
  calculations. Evaluates materials against DOE hydrogen storage targets.

  Scientific Foundation:
    Reaction:  Metal-Hydride  →  Dehydride  +  n/2 H2(g)

    ΔH(T) = [E_DFT(dehyd) + F_phonon(dehyd,T)] - [E_DFT(hyd) + F_phonon(hyd,T)]
            + (n/2) × [E_DFT(H2) + H_gas(H2,T)]

    ΔS(T) = S_vib(dehyd,T) - S_vib(hyd,T) - (n/2) × S_total(H2,T)

    ΔG(T) = ΔH(T) - T × ΔS(T)

    T_des (at P_H2): ΔG(T_des) = 0  →  ln(P) = ΔH/(RT) - ΔS/R

    wt% = (n × M_H) / M_hydride × 100
    g/L = (n × M_H) / V_molar

  References:
    [1] Bogdanovic et al., J. Alloys Compd. 382, 1 (2004) -- MgH2
    [2] Zuttel et al., Nature Mater. 4, 673 (2005) -- LiBH4
    [3] DOE Hydrogen Program Plan (2023) -- System targets
    [4] NIST Chemistry WebBook -- H2 gas properties
    [5] Baricco et al., J. Alloys Compd. 509, S344 (2011) -- Van't Hoff
=============================================================================
"""
import os
import json
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple

from .physics import R_GAS, H_PLANCK, K_BOLTZMANN, N_AVOGADRO, THZ_TO_CM
from .h2_molecule import (
    h2_total_entropy, h2_enthalpy, h2_gibbs, H2Constants
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
# CONSTANTS
# ============================================================================

M_H = 1.008           # g/mol (hydrogen atomic mass)
M_H2 = 2.01588        # g/mol (H2 molecular mass)

# DOE 2025 System Targets
DOE_GRAVIMETRIC_TARGET = 5.5      # wt%
DOE_VOLUMETRIC_TARGET = 40.0      # g/L
DOE_TDES_MIN = 300                # K (practical operating range)
DOE_TDES_MAX = 500                # K

# Hydride stretch frequency ranges (cm⁻¹)
HYDRIDE_TYPE_RANGES = {
    'ionic':         (1100, 1400),   # MgH2, CaH2, NaH
    'complex_alh':   (1700, 1850),   # NaAlH4 (Al-H stretch)
    'borohydride':   (2200, 2600),   # LiBH4, NaBH4 (B-H stretch)
    'amide':         (3100, 3500),   # LiNH2, Mg(NH2)2 (N-H stretch)
    'hydrocarbon':   (2800, 3100),   # C-H stretch (impurity)
}


# ============================================================================
# DATA CLASSES
# ============================================================================

class HStorageResult:
    """Complete hydrogen storage analysis results."""

    def __init__(self):
        # Input
        self.hydride_formula = ""
        self.dehydride_formula = ""
        self.n_h2 = 0.0            # moles H2 released per reaction

        # Thermodynamics
        self.delta_H_0K = 0.0      # kJ/mol H2
        self.delta_S_300K = 0.0    # J/(mol H2·K)
        self.delta_G_T = np.array([])  # kJ/mol H2 vs T
        self.delta_H_T = np.array([])  # kJ/mol H2 vs T
        self.delta_S_T = np.array([])  # J/(mol H2·K) vs T
        self.temperatures = np.array([])  # K

        # Desorption temperatures
        self.T_des_1bar = 0.0
        self.T_des_01bar = 0.0
        self.T_des_5bar = 0.0

        # Capacities
        self.wt_percent = 0.0
        self.g_per_liter = 0.0

        # Van't Hoff data
        self.vh_temperatures = np.array([])
        self.vh_pressures = np.array([])  # bar

        # H-vibrational analysis
        self.h_mode_fractions = {'lib': 0, 'bend': 0, 'stretch': 0}
        self.h_peak_freq_thz = 0.0
        self.h_peak_freq_cm = 0.0
        self.hydride_type = "unknown"

        # DOE evaluation
        self.doe_gravimetric_pass = False
        self.doe_volumetric_pass = False
        self.doe_tdes_pass = False

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Hydrogen Storage Analysis: {self.hydride_formula}")
        lines.append("=" * 60)
        lines.append(f"  Reaction: {self.hydride_formula} → {self.dehydride_formula} + {self.n_h2} H2")
        lines.append("")

        lines.append("  Thermodynamics:")
        lines.append(f"    ΔH (0 K)     = {self.delta_H_0K:8.1f} kJ/mol H2")
        lines.append(f"    ΔS (300 K)   = {self.delta_S_300K:8.1f} J/(mol H2·K)")
        if len(self.temperatures) > 0:
            idx_300 = int(np.argmin(np.abs(self.temperatures - 300)))
            lines.append(f"    ΔG (300 K)   = {self.delta_G_T[idx_300]:8.1f} kJ/mol H2")
        lines.append("")

        lines.append("  Desorption Temperatures:")
        lines.append(f"    T_des (1 bar)   = {self.T_des_1bar:6.0f} K  ({self.T_des_1bar - 273.15:.0f} °C)")
        lines.append(f"    T_des (0.1 bar) = {self.T_des_01bar:6.0f} K  ({self.T_des_01bar - 273.15:.0f} °C)")
        lines.append(f"    T_des (5 bar)   = {self.T_des_5bar:6.0f} K  ({self.T_des_5bar - 273.15:.0f} °C)")
        lines.append("")

        lines.append("  Capacities:")
        lines.append(f"    Gravimetric:  {self.wt_percent:6.2f} wt%")
        lines.append(f"    Volumetric:  {self.g_per_liter:6.1f} g/L")
        lines.append("")

        lines.append("  Hydrogen Vibrational Analysis:")
        fm = self.h_mode_fractions
        lines.append(f"    H-mode decomposition:")
        lines.append(f"      Librational:  {fm['lib']:5.1f} %")
        lines.append(f"      Bending:      {fm['bend']:5.1f} %")
        lines.append(f"      Stretching:   {fm['stretch']:5.1f} %")
        if self.h_peak_freq_cm > 0:
            lines.append(f"    Principal stretch: {self.h_peak_freq_cm:.0f} cm⁻¹ ({self.h_peak_freq_thz:.1f} THz)")
            lines.append(f"    Hydride type: {self.hydride_type}")
        lines.append("")

        lines.append("  DOE 2025 Target Comparison:")
        grav = "PASS ✓" if self.doe_gravimetric_pass else "FAIL ✗"
        vol = "PASS ✓" if self.doe_volumetric_pass else "FAIL ✗"
        tdes = "PASS ✓" if self.doe_tdes_pass else "WARNING: above range"
        lines.append(f"    Gravimetric:  {self.wt_percent:5.2f} wt%  vs  {DOE_GRAVIMETRIC_TARGET} wt%   [{grav}]")
        lines.append(f"    Volumetric:  {self.g_per_liter:5.1f} g/L   vs  {DOE_VOLUMETRIC_TARGET} g/L   [{vol}]")
        lines.append(f"    T_des (1 bar): {self.T_des_1bar:5.0f} K   vs  300-500 K  [{tdes}]")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'hydride_formula': self.hydride_formula,
            'dehydride_formula': self.dehydride_formula,
            'n_h2': self.n_h2,
            'delta_H_0K_kJ_mol': self.delta_H_0K,
            'delta_S_300K_J_molK': self.delta_S_300K,
            'T_des_1bar_K': self.T_des_1bar,
            'T_des_01bar_K': self.T_des_01bar,
            'T_des_5bar_K': self.T_des_5bar,
            'wt_percent': self.wt_percent,
            'g_per_liter': self.g_per_liter,
            'h_mode_fractions': self.h_mode_fractions,
            'h_peak_freq_cm': self.h_peak_freq_cm,
            'hydride_type': self.hydride_type,
            'doe_gravimetric_pass': self.doe_gravimetric_pass,
            'doe_volumetric_pass': self.doe_volumetric_pass,
            'temperatures_K': self.temperatures.tolist(),
            'delta_G_kJ_mol': self.delta_G_T.tolist(),
            'delta_H_kJ_mol': self.delta_H_T.tolist(),
            'delta_S_J_molK': self.delta_S_T.tolist(),
        }


# ============================================================================
# MAIN ENGINE
# ============================================================================

class HStorageAnalyzer:
    """Complete hydrogen storage thermodynamics analyzer.

    Workflow:
        1. Load hydride phase (DFT energy + phonon free energy + entropy)
        2. Load dehydride phase (DFT energy + phonon free energy + entropy)
        3. Compute ΔH, ΔS, ΔG vs T
        4. Find T_des from ΔG = 0 at various pressures
        5. Generate Van't Hoff plot
        6. Evaluate against DOE targets
        7. Analyze H vibrational modes
    """

    def __init__(self, hydride_data, dehydride_data, n_h2=None):
        """
        Args:
            hydride_data: dict with {
                'formula': 'MgH2',
                'e_dft': -5.67,  # eV per formula unit
                'free_energy_T': [...],  # kJ/mol vs T
                'entropy_T': [...],  # J/(mol·K) vs T
                'volume': 30.2,  # Å³ per formula unit
                'n_h': 2,  # H atoms per formula unit
                'temperatures': [0, 10, ..., 1000],
            }
            dehydride_data: dict (same structure, n_h = 0)
            n_h2: moles H2 released (auto-computed from n_h if None)
        """
        self.hydride = hydride_data
        self.dehydride = dehydride_data
        self.n_h = hydride_data.get('n_h', 0)
        self.n_h2 = n_h2 if n_h2 is not None else self.n_h / 2.0
        self.result = HStorageResult()
        self.result.hydride_formula = hydride_data.get('formula', 'Unknown')
        self.result.dehydride_formula = dehydride_data.get('formula', 'Unknown')
        self.result.n_h2 = self.n_h2

    def execute(self, t_min=0, t_max=1000, t_step=10) -> HStorageResult:
        """Run complete H-storage analysis."""
        result = self.result

        # Build temperature array
        temperatures = np.arange(t_min, t_max + t_step, t_step, dtype=float)
        result.temperatures = temperatures

        # Step 1: Compute ΔH(T)
        delta_H_T = self._compute_enthalpy(temperatures)
        result.delta_H_T = delta_H_T
        result.delta_H_0K = delta_H_T[0] if len(delta_H_T) > 0 else 0.0

        # Step 2: Compute ΔS(T)
        delta_S_T = self._compute_entropy(temperatures)
        result.delta_S_T = delta_S_T
        result.delta_S_300K = np.interp(300, temperatures, delta_S_T)

        # Step 3: Compute ΔG(T)
        delta_G_T = delta_H_T - temperatures * delta_S_T / 1000  # J → kJ
        result.delta_G_T = delta_G_T

        # Step 4: Find desorption temperatures
        result.T_des_1bar = self._find_desorption_temp(delta_H_T, delta_S_T, temperatures, 1.0)
        result.T_des_01bar = self._find_desorption_temp(delta_H_T, delta_S_T, temperatures, 0.1)
        result.T_des_5bar = self._find_desorption_temp(delta_H_T, delta_S_T, temperatures, 5.0)

        # Step 5: Compute capacities
        self._compute_capacities()

        # Step 6: Van't Hoff data
        self._generate_vant_hoff(delta_H_T[0], delta_S_T[0] if len(delta_S_T) > 0 else 130.0)

        # Step 7: DOE targets
        self._evaluate_doe_targets()

        return result

    def _compute_enthalpy(self, temperatures):
        """Compute ΔH_des(T) in kJ/mol H2.

        ΔH(T) = [E_DFT(dehyd) + F_phonon(dehyd,T) - T×S_vib(dehyd,T)]
                - [E_DFT(hyd) + F_phonon(hyd,T) - T×S_vib(hyd,T)]
                + (n_H2/2) × [E_DFT(H2) + H_gas(H2,T)]

        Note: F_phonon = E_DFT + F_vib, so F_phonon - T×S_vib = E_DFT + U_vib - T×S_vib
            = E_DFT + F_vib = G_vib

        Simplified: ΔH(T) = ΔE_DFT + ΔH_vib(T) + (n_H2/2) × H_gas(H2,T)

        Args:
            temperatures: K

        Returns:
            ΔH vs T in kJ/mol H2
        """
        # DFT energy difference (eV → kJ/mol: multiply by 96.485)
        EV_TO_KJ_MOL = 96.48533212
        e_hyd = self.hydride.get('e_dft', 0.0)
        e_dehyd = self.dehydride.get('e_dft', 0.0)

        # H2 DFT energy (reference: -6.77 eV for isolated H2 molecule)
        e_h2 = self.hydride.get('h2_energy', -6.77)

        # Electronic enthalpy change (0 K)
        delta_E_elec = (e_dehyd + self.n_h2 * e_h2 - e_hyd) * EV_TO_KJ_MOL  # kJ per reaction
        delta_E_elec_per_h2 = delta_E_elec / self.n_h2  # kJ/mol H2

        # Vibrational enthalpy contribution
        # F = U - TS → U = F + TS
        # ΔH_vib = U_vib(dehyd) - U_vib(hyd)
        F_hyd = np.array(self.hydride.get('free_energy_T', [0.0] * len(temperatures)))
        F_dehyd = np.array(self.dehydride.get('free_energy_T', [0.0] * len(temperatures)))
        S_hyd = np.array(self.hydride.get('entropy_T', [0.0] * len(temperatures)))
        S_dehyd = np.array(self.dehydride.get('entropy_T', [0.0] * len(temperatures)))

        # U_vib = F_vib + T*S_vib (vibrational internal energy)
        # For T=0, U_vib = ZPE
        U_hyd = F_hyd + temperatures * S_hyd / 1000  # kJ/mol
        U_dehyd = F_dehyd + temperatures * S_dehyd / 1000

        delta_U_vib = (U_dehyd - U_hyd) / self.n_h2  # kJ/mol H2

        # H2 gas enthalpy (J/mol → kJ/mol H2)
        H_h2_gas = np.array([h2_enthalpy(T) / 1000 for T in temperatures])  # kJ/mol H2

        delta_H_T = delta_E_elec_per_h2 + delta_U_vib + H_h2_gas

        return delta_H_T

    def _compute_entropy(self, temperatures):
        """Compute ΔS_des(T) in J/(mol H2·K).

        ΔS(T) = S_vib(dehyd,T) - S_vib(hyd,T) - (n_H2/2) × S_total(H2,T)

        The dominant term is S_total(H2) which includes translational,
        rotational, and vibrational entropy of H2 gas.

        At 300K, 1 bar:
            S_total(H2) ≈ 130.7 J/(mol·K)
            For n_H2 = 1: ΔS ≈ -130.7 J/(mol H2·K) (plus small vib corrections)

        Args:
            temperatures: K

        Returns:
            ΔS vs T in J/(mol H2·K)
        """
        S_hyd = np.array(self.hydride.get('entropy_T', [0.0] * len(temperatures)))
        S_dehyd = np.array(self.dehydride.get('entropy_T', [0.0] * len(temperatures)))

        # H2 gas entropy at 1 bar
        S_h2_gas = np.array([h2_total_entropy(T, 1e5) for T in temperatures])  # J/(mol·K)

        # ΔS per mol H2
        delta_S_T = (S_dehyd - S_hyd - self.n_h2 * S_h2_gas) / self.n_h2

        return delta_S_T

    def _find_desorption_temp(self, delta_H_T, delta_S_T, temperatures, P_H2_bar):
        """Find T where ΔG(T, P_H2) = 0.

        From Van't Hoff: ln(P/P0) = ΔH/(RT) - ΔS/R
        At equilibrium: ΔG = ΔH - TΔS + RT×ln(P/P0) = 0
        → T_des = ΔH / (ΔS - R×ln(P/P0))

        For P0 = 1 bar: T_des = ΔH / (ΔS - R×ln(P))

        Args:
            delta_H_T: kJ/mol H2
            delta_S_T: J/(mol H2·K)
            temperatures: K
            P_H2_bar: equilibrium H2 pressure (bar)

        Returns:
            T_des in K
        """
        # Use average values over temperature range
        # For simplicity, use values at mid-range temperature
        idx_mid = len(temperatures) // 2
        delta_H_avg = delta_H_T[idx_mid] * 1000  # kJ → J
        delta_S_avg = delta_S_T[idx_mid]

        if delta_S_avg <= 0 or delta_H_avg <= 0:
            return 0.0

        # Van't Hoff correction for non-1-bar pressure
        R = R_GAS  # J/(mol·K)
        delta_S_corrected = delta_S_avg - R * np.log(P_H2_bar)

        if delta_S_corrected <= 0:
            return 0.0

        T_des = delta_H_avg / delta_S_corrected

        # Validate: T_des should be within reasonable range
        if T_des < 100 or T_des > 1500:
            # Fallback: find where ΔG crosses zero
            delta_G_T = delta_H_T - temperatures * delta_S_T / 1000
            # Find sign change
            for i in range(1, len(delta_G_T)):
                if delta_G_T[i-1] > 0 and delta_G_T[i] < 0:
                    # Linear interpolation
                    T_des = temperatures[i-1] + (temperatures[i] - temperatures[i-1]) * \
                            delta_G_T[i-1] / (delta_G_T[i-1] - delta_G_T[i])
                    break

        return T_des

    def _compute_capacities(self):
        """Compute gravimetric and volumetric hydrogen capacities.

        wt% = (n_H × M_H) / M_hydride × 100
        g/L = (n_H × M_H) / V_hydride

        Where M_hydride = molar mass of hydride formula unit
              V_hydride = volume per formula unit
        """
        n_h = self.n_h
        formula = self.hydride.get('formula', '')
        volume = self.hydride.get('volume', 0.0)  # Å³ per formula unit

        # Estimate molar mass from formula (simplified parsing)
        M_hydride = self._estimate_molar_mass(formula)
        if M_hydride <= 0:
            M_hydride = 100.0  # fallback

        # Gravimetric capacity
        self.result.wt_percent = (n_h * M_H) / M_hydride * 100

        # Volumetric capacity
        if volume > 0:
            # Volume per formula unit in cm³/mol: V(Å³) × 1e-24 × N_A
            V_cm3_mol = volume * 1e-24 * N_AVOGADRO
            # g H2 per liter: (n_H × M_H) / V_cm3_mol × 1000
            self.result.g_per_liter = (n_h * M_H) / V_cm3_mol * 1000
        else:
            self.result.g_per_liter = 0.0

    def _estimate_molar_mass(self, formula):
        """Estimate molar mass from chemical formula.

        Simple parser: handles formulas like MgH2, LiBH4, NaAlH4, etc.
        """
        from .analyzer import ELEMENT_DB
        import re

        total = 0.0
        # Parse formula: element followed by optional number
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)

        for elem, count in matches:
            if elem in ELEMENT_DB:
                mass = ELEMENT_DB[elem][1]  # amu = g/mol
                n = int(count) if count else 1
                total += mass * n

        return total

    def _generate_vant_hoff(self, delta_H, delta_S_300K):
        """Generate Van't Hoff plot data.

        ln(P_H2/bar) = ΔH/(R×T) - ΔS/R

        Uses constant ΔH, ΔS approximation (standard for Van't Hoff plots).

        Args:
            delta_H: kJ/mol H2 (assumed T-independent)
            delta_S_300K: J/(mol H2·K) (assumed T-independent)
        """
        delta_H_J = delta_H * 1000  # J/mol
        R = R_GAS

        T_range = np.linspace(300, 800, 100)
        ln_P = delta_H_J / (R * T_range) - delta_S_300K / R
        P_bar = np.exp(ln_P)

        # Clip to reasonable range
        P_bar = np.clip(P_bar, 1e-6, 1e6)

        self.result.vh_temperatures = T_range
        self.result.vh_pressures = P_bar

    def _evaluate_doe_targets(self):
        """Evaluate against DOE 2025 system targets."""
        self.result.doe_gravimetric_pass = self.result.wt_percent >= DOE_GRAVIMETRIC_TARGET
        self.result.doe_volumetric_pass = self.result.g_per_liter >= DOE_VOLUMETRIC_TARGET
        self.result.doe_tdes_pass = (DOE_TDES_MIN <= self.result.T_des_1bar <= DOE_TDES_MAX)

    def analyze_hydrogen_modes(self, h_dos, freq_thz):
        """Analyze hydrogen vibrational mode decomposition.

        Args:
            h_dos: H partial DOS array (states/THz)
            freq_thz: frequency array (THz)
        """
        if len(h_dos) == 0 or len(freq_thz) == 0:
            return

        # Integration ranges from physics module
        lib_range = (5.0, 20.9)
        bend_range = (20.9, 50.0)
        stretch_range = (50.0, 100.0)

        lib_mask = (freq_thz > lib_range[0]) & (freq_thz < lib_range[1])
        bend_mask = (freq_thz > bend_range[0]) & (freq_thz < bend_range[1])
        stretch_mask = (freq_thz > stretch_range[0]) & (freq_thz < stretch_range[1])

        h_total = np.trapz(h_dos[freq_thz > 0], freq_thz[freq_thz > 0])
        h_lib = np.trapz(h_dos[lib_mask], freq_thz[lib_mask]) if np.any(lib_mask) else 0.0
        h_bend = np.trapz(h_dos[bend_mask], freq_thz[bend_mask]) if np.any(bend_mask) else 0.0
        h_stretch = np.trapz(h_dos[stretch_mask], freq_thz[stretch_mask]) if np.any(stretch_mask) else 0.0

        if h_total > 0:
            self.result.h_mode_fractions = {
                'lib': h_lib / h_total * 100,
                'bend': h_bend / h_total * 100,
                'stretch': h_stretch / h_total * 100,
            }

            # Find peak stretching frequency
            if np.any(stretch_mask):
                peak_idx = np.argmax(h_dos[stretch_mask])
                peak_thz = freq_thz[stretch_mask][peak_idx]
                peak_cm = peak_thz * THZ_TO_CM
                self.result.h_peak_freq_thz = peak_thz
                self.result.h_peak_freq_cm = peak_cm

                # Classify hydride type
                self.result.hydride_type = self._classify_hydride(peak_cm)

    def _classify_hydride(self, stretch_freq_cm):
        """Identify hydride type from stretch frequency."""
        for hydride_type, (low, high) in HYDRIDE_TYPE_RANGES.items():
            if low <= stretch_freq_cm <= high:
                return hydride_type
        return "unknown"

    def plot_results(self, output_dir):
        """Generate publication-quality H-storage plots."""
        os.makedirs(output_dir, exist_ok=True)
        r = self.result

        # 1. Thermodynamics: ΔH, ΔS, ΔG vs T
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(r.temperatures, r.delta_G_T, 'r-', lw=2, label=r'$\Delta$G')
        ax1.plot(r.temperatures, r.delta_H_T, 'b-', lw=2, label=r'$\Delta$H')
        ax1.axhline(y=0, color='gray', ls='--', alpha=0.5)
        ax1.set_xlabel('T (K)')
        ax1.set_ylabel('Energy (kJ/mol H$_2$)')
        ax1.set_title('Dehydrogenation Thermodynamics')
        ax1.legend()
        ax1.grid(alpha=0.2)

        ax2.plot(r.temperatures, r.delta_S_T, 'g-', lw=2)
        ax2.set_xlabel('T (K)')
        ax2.set_ylabel(r'$\Delta$S (J/mol$\cdot$K)')
        ax2.set_title('Entropy Change')
        ax2.grid(alpha=0.2)

        plt.tight_layout()
        path = os.path.join(output_dir, 'h_thermodynamics.png')
        plt.savefig(path)
        plt.close()
        print(f"  [H-Storage] Saved: {path}")

        # 2. Van't Hoff plot
        if len(r.vh_temperatures) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(1000 / r.vh_temperatures, np.log(r.vh_pressures), 'b-', lw=2)
            ax.set_xlabel('1000/T (K$^{-1}$)')
            ax.set_ylabel('ln(P$_{H_2}$ / bar)')
            ax.set_title(f'Van\'t Hoff Plot: {r.hydride_formula}')
            ax.grid(alpha=0.2)
            plt.tight_layout()
            path = os.path.join(output_dir, 'vant_hoff.png')
            plt.savefig(path)
            plt.close()
            print(f"  [H-Storage] Saved: {path}")

        # Save data
        data_path = os.path.join(output_dir, 'h_storage_data.json')
        with open(data_path, 'w') as f:
            json.dump(r.to_dict(), f, indent=2)
        print(f"  [H-Storage] Saved: {data_path}")
