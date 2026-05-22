"""
=============================================================================
  HydroPhonoKit Visualization — Thermodynamic Plots

  Implements all thermodynamic plot types for phonon analysis:
    1. F(T), S(T), Cv(T) Triple Panel - Complete thermodynamic overview
    2. Cv vs T with Dulong-Petit - Heat capacity validation
    3. Free Energy Components - ZPE + thermal contributions
    4. Entropy Decomposition - Per-element entropy contributions
    5. Cp vs Cv Comparison - Constant pressure vs volume
    6. Temperature-Dependent Properties - All properties vs T
    7. Low-T Behavior (T³ law) - Debye model validation
    8. High-T Dulong-Petit Check - Convergence verification

  Scientific Foundation:
    Phonon thermodynamic properties are computed from the phonon DOS g(omega)
    using the harmonic approximation:

    Free Energy:
      F(T) = k_B T ∫ dω g(ω) ln[2 sinh(ℏω/2k_B T)]

    Entropy:
      S(T) = -∂F/∂T = k_B ∫ dω g(ω) [x coth(x/2) - ln(2 sinh(x/2))]
      where x = ℏω/k_B T

    Heat Capacity (constant volume):
      C_v(T) = -T ∂²F/∂T² = k_B ∫ dω g(ω) x²/(4 sinh²(x/2))

    Heat Capacity (constant pressure):
      C_p(T) = C_v(T) + T V α² B
      where α = thermal expansion coefficient, B = bulk modulus

    Key validation criteria:
      - Third Law: S(T→0) → 0
      - Dulong-Petit: C_v(T→∞) → 3N k_B
      - Low-T: C_v ∝ T³ (Debye law)
      - High-T: C_p ≈ C_v (for solids)

  References:
    [1] Born & Huang, Dynamical Theory of Crystal Lattices (1954)
    [2] Grimvall, Thermophysical Properties of Materials (1999)
    [3] Togo et al., Phys. Rev. B 78, 134306 (2008) -- Phonopy thermo
    [4] Debye, Ann. Phys. 39, 789 (1912) -- T³ law
=============================================================================
"""
import os
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

# Configure matplotlib backend
import os as _os
if not _os.environ.get('MPLBACKEND'):
    import matplotlib
    matplotlib.use('Agg')
del _os

import matplotlib.pyplot as plt
import matplotlib.axes as mpl_axes
from matplotlib.figure import Figure

from .base_plotter import BasePlotter
from .themes import get_element_color, ELEMENT_COLORS


# ============================================================================
# CONSTANTS
# ============================================================================

R_GAS = 8.314462618  # J/(mol·K) - universal gas constant
KB = 1.380649e-23    # J/K - Boltzmann constant
NA = 6.02214076e23   # 1/mol - Avogadro's number


# ============================================================================
# THERMODYNAMIC PLOTTER CLASS
# ============================================================================

class ThermoPlotter(BasePlotter):
    """Specialized plotter for thermodynamic properties.
    
    Provides methods for all thermodynamic plot types.
    """
    
    def plot_triple_panel(self, temps: np.ndarray,
                         free_energy: np.ndarray,
                         entropy: np.ndarray,
                         heat_capacity: np.ndarray,
                         formula: str = 'Material',
                         n_atoms: int = None,
                         show_dulong_petit: bool = True,
                         fig: Figure = None,
                         axes: Tuple = None,
                         figsize: Tuple[float, float] = (14, 5),
                         title: str = None) -> Tuple[Figure, Tuple]:
        """Plot F(T), S(T), Cv(T) in triple panel layout.
        
        This is the standard thermodynamic overview plot showing all three
        fundamental thermodynamic quantities.
        
        Args:
            temps: Temperature array (K)
            free_energy: Helmholtz free energy F(T) (kJ/mol)
            entropy: Entropy S(T) (J/(mol·K))
            heat_capacity: Heat capacity Cv(T) (J/(mol·K))
            formula: Material formula
            n_atoms: Number of atoms per unit cell (for Dulong-Petit)
            show_dulong_petit: Show Dulong-Petit limit on Cv plot
            fig: Existing figure (creates new if None)
            axes: Existing axes tuple (ax1, ax2, ax3)
            figsize: Figure size if creating new
            title: Custom title
        
        Returns:
            (figure, (ax1, ax2, ax3)) tuple
        """
        if fig is None or axes is None:
            fig, (ax1, ax2, ax3) = self.create_figure(
                nrows=1, ncols=3, figsize=figsize)
        else:
            ax1, ax2, ax3 = axes
        
        # Free Energy F(T)
        ax1.plot(temps, free_energy, color=self.get_color('primary'), lw=2)
        ax1.set_xlabel('T (K)')
        ax1.set_ylabel('F (kJ/mol)')
        ax1.set_title('F(T)', fontweight='bold')
        ax1.grid(alpha=self.theme['axes']['grid_alpha'])
        
        # Mark ZPE at T=0
        if len(free_energy) > 0:
            ax1.axhline(y=free_energy[0], color=self.get_color('tertiary'),
                       ls=':', alpha=0.5)
            ax1.text(0.02, 0.95, f'ZPE: {free_energy[0]:.2f} kJ/mol',
                    transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Entropy S(T)
        ax2.plot(temps, entropy, color=self.get_color('secondary'), lw=2)
        ax2.set_xlabel('T (K)')
        ax2.set_ylabel('S (J/(mol·K))')
        ax2.set_title('S(T)', fontweight='bold')
        ax2.grid(alpha=self.theme['axes']['grid_alpha'])
        
        # Third Law validation
        if len(entropy) > 0 and abs(entropy[0]) < 0.01:
            ax2.text(0.02, 0.95, 'Third Law: OK',
                    transform=ax2.transAxes, fontsize=8, color='green',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='#D1FAE5', alpha=0.8))
        
        # Heat Capacity Cv(T)
        ax3.plot(temps, heat_capacity, color=self.get_color('tertiary'), lw=2)
        
        if show_dulong_petit and n_atoms:
            dp_limit = 3 * n_atoms * R_GAS
            ax3.axhline(y=dp_limit, color='gray', ls='--', alpha=0.5,
                       label=f'Dulong-Petit = {dp_limit:.0f}')
            ax3.legend(fontsize=9)
        
        ax3.set_xlabel('T (K)')
        ax3.set_ylabel('Cv (J/(mol·K))')
        ax3.set_title('Cv(T)', fontweight='bold')
        ax3.grid(alpha=self.theme['axes']['grid_alpha'])
        
        # Overall title
        if title is None:
            title = f'{formula} - Vibrational Thermodynamics'
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, (ax1, ax2, ax3)
    
    def plot_cv_dulong_petit(self, temps: np.ndarray,
                            heat_capacity: np.ndarray,
                            formula: str = 'Material',
                            n_atoms: int = None,
                            show_deviation: bool = True,
                            ax: mpl_axes.Axes = None,
                            figsize: Tuple[float, float] = (8, 6),
                            title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot Cv(T) with Dulong-Petit limit and deviation analysis.
        
        Focuses on validating the high-temperature behavior of heat capacity.
        
        Args:
            temps: Temperature array (K)
            heat_capacity: Heat capacity Cv(T) (J/(mol·K))
            formula: Material formula
            n_atoms: Number of atoms per unit cell
            show_deviation: Show deviation from Dulong-Petit as inset
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        if n_atoms is None:
            n_atoms = 1
        
        dp_limit = 3 * n_atoms * R_GAS
        
        # Main Cv plot
        ax.plot(temps, heat_capacity, color=self.get_color('primary'), lw=2,
               label='Cv')
        ax.axhline(y=dp_limit, color='gray', ls='--', alpha=0.7,
                  label=f'Dulong-Petit (3Nk$_B$) = {dp_limit:.1f}')
        
        # Mark convergence point
        if len(heat_capacity) > 0:
            # Find where Cv reaches 95% of Dulong-Petit
            threshold = 0.95 * dp_limit
            conv_indices = np.where(heat_capacity >= threshold)[0]
            if len(conv_indices) > 0:
                conv_temp = temps[conv_indices[0]]
                ax.axvline(x=conv_temp, color=self.get_color('secondary'),
                          ls=':', alpha=0.7)
                ax.text(conv_temp, dp_limit * 0.5,
                       f'95% convergence: {conv_temp:.0f} K',
                       color=self.get_color('secondary'), fontsize=9,
                       rotation=90, va='center')
        
        # Deviation inset
        if show_deviation and len(heat_capacity) > 1:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_inset = inset_axes(ax, width="40%", height="30%", loc='lower right',
                                 borderpad=1)
            
            deviation = (heat_capacity - dp_limit) / dp_limit * 100
            ax_inset.plot(temps, deviation, color=self.get_color('secondary'), lw=1.5)
            ax_inset.axhline(y=0, color='gray', ls='--', alpha=0.5)
            ax_inset.set_xlabel('T (K)', fontsize=7)
            ax_inset.set_ylabel('Deviation (%)', fontsize=7)
            ax_inset.set_title('D-P Deviation', fontsize=8)
            ax_inset.tick_params(labelsize=6)
            ax_inset.grid(alpha=0.3)
        
        # Final deviation annotation
        if len(heat_capacity) > 0:
            final_dev = (heat_capacity[-1] - dp_limit) / dp_limit * 100
            ax.text(0.98, 0.02, f'@ {temps[-1]:.0f} K: {final_dev:+.1f}%',
                   transform=ax.transAxes, fontsize=9, ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('Cv (J/(mol·K))')
        ax.legend()
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Cv(T) with Dulong-Petit Validation'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_free_energy_components(self, temps: np.ndarray,
                                   free_energy: np.ndarray,
                                   entropy: np.ndarray,
                                   formula: str = 'Material',
                                   ax: mpl_axes.Axes = None,
                                   figsize: Tuple[float, float] = (8, 6),
                                   title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot free energy decomposition into components.
        
        Shows:
          - F(T) = U(T) - T*S(T)
          - Zero-point energy (ZPE)
          - Thermal contribution
          - -T*S(T) term
        
        Args:
            temps: Temperature array (K)
            free_energy: Helmholtz free energy F(T) (kJ/mol)
            entropy: Entropy S(T) (J/(mol·K))
            formula: Material formula
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        # Calculate components
        zpe = free_energy[0] if len(free_energy) > 0 else 0
        ts_term = -temps * entropy / 1000  # Convert J to kJ
        thermal_contrib = free_energy - zpe
        
        # Plot components
        ax.fill_between(temps, zpe, free_energy, alpha=0.3,
                       color=self.get_color('secondary'),
                       label='-T·S(T) (entropic)')
        ax.plot(temps, free_energy, color=self.get_color('primary'), lw=2,
               label='F(T) = U - T·S')
        ax.axhline(y=zpe, color=self.get_color('tertiary'), ls='--', alpha=0.7,
                  label=f'ZPE = {zpe:.2f} kJ/mol')
        
        # Annotations
        if len(temps) > 1:
            ts_final = ts_term[-1]
            ax.text(0.98, 0.02,
                   f'-T·S(@{temps[-1]:.0f}K) = {ts_final:.2f} kJ/mol',
                   transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('Energy (kJ/mol)')
        ax.legend()
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Free Energy Components'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_entropy_decomposition(self, temps: np.ndarray,
                                  entropy: np.ndarray,
                                  element_entropies: Dict[str, np.ndarray],
                                  formula: str = 'Material',
                                  stacked: bool = True,
                                  ax: mpl_axes.Axes = None,
                                  figsize: Tuple[float, float] = (8, 6),
                                  title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot entropy decomposition by element.
        
        Shows each element's contribution to total entropy.
        
        Args:
            temps: Temperature array (K)
            entropy: Total entropy S(T) (J/(mol·K))
            element_entropies: Dict mapping element to S_i(T) array
            formula: Material formula
            stacked: Stack element contributions (True) or overlay (False)
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        if stacked:
            # Stacked area plot
            elements = list(element_entropies.keys())
            entropy_components = [element_entropies[e] for e in elements]
            colors = [self.get_element_color(e) for e in elements]
            
            ax.stackplot(temps, entropy_components, labels=elements,
                        colors=colors, alpha=0.7)
        else:
            # Overlay plot
            for elem, s_elem in element_entropies.items():
                c = self.get_element_color(elem)
                ax.fill_between(temps, s_elem, alpha=0.2, color=c)
                ax.plot(temps, s_elem, color=c, lw=1.5, label=elem)
        
        # Total entropy line
        ax.plot(temps, entropy, color='black', lw=2, ls='--',
               alpha=0.7, label='Total')
        
        # Percentage at high T
        if len(temps) > 0:
            idx = -1
            percentages = []
            for elem in element_entropies:
                if entropy[idx] > 0:
                    pct = element_entropies[elem][idx] / entropy[idx] * 100
                    percentages.append(f'{elem}: {pct:.1f}%')
            
            if percentages:
                ax.text(0.02, 0.98, '@ ' + f'{temps[idx]:.0f}K: ' + ', '.join(percentages),
                       transform=ax.transAxes, fontsize=8, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('S (J/(mol·K))')
        ax.legend(loc='upper left')
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Entropy Decomposition'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_cp_vs_cv(self, temps: np.ndarray,
                     cv: np.ndarray,
                     cp: np.ndarray,
                     formula: str = 'Material',
                     show_difference: bool = True,
                     ax: mpl_axes.Axes = None,
                     figsize: Tuple[float, float] = (8, 6),
                     title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot Cp vs Cv comparison.
        
        Cp = Cv + T*V*α²*B
        
        Args:
            temps: Temperature array (K)
            cv: Cv(T) (J/(mol·K))
            cp: Cp(T) (J/(mol·K))
            formula: Material formula
            show_difference: Show Cp - Cv as inset
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot both curves
        ax.plot(temps, cv, color=self.get_color('primary'), lw=2, label='Cv')
        ax.plot(temps, cp, color=self.get_color('secondary'), lw=2, label='Cp')
        
        # Difference annotation
        if len(cp) > 0 and len(cv) > 0:
            diff = cp[-1] - cv[-1]
            ax.text(0.98, 0.02, f'@ {temps[-1]:.0f}K: Cp - Cv = {diff:.2f} J/(mol·K)',
                   transform=ax.transAxes, fontsize=9, ha='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Difference inset
        if show_difference and len(cp) > 1:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_inset = inset_axes(ax, width="40%", height="30%", loc='upper left',
                                 borderpad=1)
            
            diff = cp - cv
            ax_inset.plot(temps, diff, color=self.get_color('tertiary'), lw=1.5)
            ax_inset.set_xlabel('T (K)', fontsize=7)
            ax_inset.set_ylabel('Cp - Cv', fontsize=7)
            ax_inset.set_title('Difference', fontsize=8)
            ax_inset.tick_params(labelsize=6)
            ax_inset.grid(alpha=0.3)
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('C (J/(mol·K))')
        ax.legend()
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - Cp vs Cv Comparison'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_all_properties(self, temps: np.ndarray,
                           free_energy: np.ndarray,
                           entropy: np.ndarray,
                           cv: np.ndarray,
                           cp: np.ndarray = None,
                           thermal_expansion: np.ndarray = None,
                           bulk_modulus: np.ndarray = None,
                           formula: str = 'Material',
                           fig: Figure = None,
                           axes: Tuple = None,
                           figsize: Tuple[float, float] = (16, 10),
                           title: str = None) -> Tuple[Figure, Tuple]:
        """Plot all thermodynamic properties in multi-panel layout.
        
        Comprehensive overview of all temperature-dependent properties.
        
        Args:
            temps: Temperature array (K)
            free_energy: F(T) (kJ/mol)
            entropy: S(T) (J/(mol·K))
            cv: Cv(T) (J/(mol·K))
            cp: Cp(T) (J/(mol·K)), optional
            thermal_expansion: α(T) (1/K), optional
            bulk_modulus: B(T) (GPa), optional
            formula: Material formula
            fig: Existing figure
            axes: Existing axes tuple
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axes_tuple) tuple
        """
        if fig is None or axes is None:
            fig, axes = plt.subplots(3, 2, figsize=figsize)
            self._figures.append(fig)
        
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
        ax5, ax6 = axes[2]
        
        # F(T)
        ax1.plot(temps, free_energy, color=self.get_color('primary'), lw=2)
        ax1.set_ylabel('F (kJ/mol)')
        ax1.set_title('Free Energy')
        ax1.grid(alpha=self.theme['axes']['grid_alpha'])
        
        # S(T)
        ax2.plot(temps, entropy, color=self.get_color('secondary'), lw=2)
        ax2.set_ylabel('S (J/(mol·K))')
        ax2.set_title('Entropy')
        ax2.grid(alpha=self.theme['axes']['grid_alpha'])
        
        # Cv(T) and Cp(T)
        ax3.plot(temps, cv, color=self.get_color('primary'), lw=2, label='Cv')
        if cp is not None:
            ax3.plot(temps, cp, color=self.get_color('secondary'), lw=2,
                    label='Cp')
            ax3.legend()
        ax3.set_ylabel('C (J/(mol·K))')
        ax3.set_title('Heat Capacity')
        ax3.grid(alpha=self.theme['axes']['grid_alpha'])
        
        # Thermal expansion
        if thermal_expansion is not None:
            ax4.plot(temps, thermal_expansion, color=self.get_color('tertiary'), lw=2)
            ax4.set_ylabel('α (1/K)')
            ax4.set_title('Thermal Expansion')
            ax4.grid(alpha=self.theme['axes']['grid_alpha'])
        else:
            ax4.set_visible(False)
        
        # Bulk modulus
        if bulk_modulus is not None:
            ax5.plot(temps, bulk_modulus, color=self.get_color('quaternary'), lw=2)
            ax5.set_ylabel('B (GPa)')
            ax5.set_title('Bulk Modulus')
            ax5.grid(alpha=self.theme['axes']['grid_alpha'])
        else:
            ax5.set_visible(False)
        
        # Cp - Cv difference
        if cp is not None:
            ax6.plot(temps, cp - cv, color=self.get_color('secondary'), lw=2)
            ax6.set_ylabel('Cp - Cv (J/(mol·K))')
            ax6.set_title('Cp - Cv Difference')
            ax6.grid(alpha=self.theme['axes']['grid_alpha'])
        else:
            ax6.set_visible(False)
        
        # Common x-labels
        for ax in axes.flatten():
            if ax.get_visible():
                ax.set_xlabel('T (K)')
        
        if title is None:
            title = f'{formula} - Complete Thermodynamic Properties'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.01)
        
        fig.tight_layout()
        
        return fig, axes
    
    def plot_low_t_behavior(self, temps: np.ndarray,
                           cv: np.ndarray,
                           formula: str = 'Material',
                           debye_temp: float = None,
                           ax: mpl_axes.Axes = None,
                           figsize: Tuple[float, float] = (8, 6),
                           title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot low-temperature Cv behavior validating T³ law.
        
        At low T: Cv ∝ T³ (Debye law)
        
        Args:
            temps: Temperature array (K)
            cv: Cv(T) (J/(mol·K))
            formula: Material formula
            debye_temp: Debye temperature for T³ fit
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        # Filter low T data (T < 100 K or first 20% of data)
        low_t_mask = temps < 100
        if np.sum(low_t_mask) < 5:
            low_t_mask = temps < temps[-1] * 0.2
        
        t_low = temps[low_t_mask]
        cv_low = cv[low_t_mask]
        
        # Plot actual data
        ax.plot(t_low, cv_low, 'o', color=self.get_color('primary'),
               markersize=4, label='Cv (calculated)')
        
        # T³ fit
        if debye_temp is not None and len(t_low) > 2:
            # Cv = (12π⁴/5) N k_B (T/Θ_D)³
            from .base_plotter import R_GAS
            n_atoms = 1  # Per formula unit
            coeff = (12 * np.pi**4 / 5) * n_atoms * R_GAS / debye_temp**3
            t_fit = np.linspace(0, max(t_low), 100)
            cv_fit = coeff * t_fit**3
            
            ax.plot(t_fit, cv_fit, '--', color=self.get_color('secondary'),
                   lw=2, label=f'T³ law (Θ_D={debye_temp:.0f}K)')
            
            # Goodness of fit
            if len(t_low) > 2:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    t_low**3, cv_low)
                ax.text(0.02, 0.95, f'R² = {r_value**2:.4f}',
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('Cv (J/(mol·K))')
        ax.legend()
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        ax.set_xlim(0, max(t_low) * 1.1)
        
        if title is None:
            title = f'{formula} - Low-T Cv Behavior (T³ Law)'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_high_t_dulong_petit(self, temps: np.ndarray,
                                cv: np.ndarray,
                                formula: str = 'Material',
                                n_atoms: int = None,
                                ax: mpl_axes.Axes = None,
                                figsize: Tuple[float, float] = (8, 6),
                                title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot high-temperature convergence to Dulong-Petit limit.
        
        At high T: Cv → 3N k_B (Dulong-Petit law)
        
        Args:
            temps: Temperature array (K)
            cv: Cv(T) (J/(mol·K))
            formula: Material formula
            n_atoms: Number of atoms per unit cell
            ax: Existing axis
            figsize: Figure size
            title: Custom title
        
        Returns:
            (figure, axis) tuple
        """
        if ax is None:
            fig, ax = self.create_figure(figsize=figsize)
        else:
            fig = ax.figure
        
        if n_atoms is None:
            n_atoms = 1
        
        dp_limit = 3 * n_atoms * R_GAS
        
        # Filter high T data
        high_t_mask = temps > 300
        t_high = temps[high_t_mask]
        cv_high = cv[high_t_mask]
        
        # Plot data
        ax.plot(t_high, cv_high, 'o', color=self.get_color('primary'),
               markersize=4, label='Cv (calculated)')
        
        # Dulong-Petit line
        ax.axhline(y=dp_limit, color=self.get_color('secondary'), lw=2,
                  ls='--', label=f'Dulong-Petit = {dp_limit:.1f}')
        
        # Convergence analysis
        if len(cv_high) > 0:
            convergence_pct = cv_high[-1] / dp_limit * 100
            deviation = (cv_high[-1] - dp_limit) / dp_limit * 100
            
            ax.text(0.02, 0.95,
                   f'@ {t_high[-1]:.0f}K: {convergence_pct:.1f}% of D-P\n'
                   f'Deviation: {deviation:+.1f}%',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Color code based on convergence
            if abs(deviation) < 5:
                status = 'GOOD'
                color = 'green'
            elif abs(deviation) < 10:
                status = 'ACCEPTABLE'
                color = 'orange'
            else:
                status = 'POOR'
                color = 'red'
            
            ax.text(0.98, 0.02, f'Convergence: {status}',
                   transform=ax.transAxes, fontsize=9, ha='right',
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('T (K)')
        ax.set_ylabel('Cv (J/(mol·K))')
        ax.legend()
        ax.grid(alpha=self.theme['axes']['grid_alpha'])
        
        if title is None:
            title = f'{formula} - High-T Dulong-Petit Convergence'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
