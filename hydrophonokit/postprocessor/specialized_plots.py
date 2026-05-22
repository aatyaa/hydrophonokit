"""
=============================================================================
  HydroPhonoKit Postprocessor — Specialized Plots

  Additional specialized plot types:
    - Cumulative thermal conductivity vs mean free path
    - Phonon convergence plots (mesh density, supercell size)
    - Multi-material overlay for comparison
=============================================================================
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

import matplotlib
if not os.environ.get('MPLBACKEND'):
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class SpecializedPlots:
    """Specialized phonon visualization plots."""

    @staticmethod
    def plot_cumulative_kappa(mean_free_paths: np.ndarray,
                            kappa_contributions: np.ndarray,
                            formula: str,
                            ax=None) -> Tuple:
        """Plot cumulative thermal conductivity vs mean free path.

        Shows what fraction of κ comes from phonons with ℓ < x.
        Essential for understanding nanostructuring effects on κ.

        Args:
            mean_free_paths: Array of mean free paths (nm), shape (n_modes,)
            kappa_contributions: κ contribution per mode (W/m·K), shape (n_modes,)
            formula: Chemical formula
            ax: Matplotlib axis

        Returns:
            (fig, ax) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        # Sort by MFP
        sort_idx = np.argsort(mean_free_paths)
        mfp_sorted = mean_free_paths[sort_idx]
        kappa_sorted = kappa_contributions[sort_idx]

        # Cumulative sum
        kappa_cum = np.cumsum(kappa_sorted)
        total_kappa = kappa_cum[-1] if kappa_cum.size > 0 else 0

        if total_kappa > 0:
            kappa_frac = kappa_cum / total_kappa * 100
        else:
            kappa_frac = np.zeros_like(kappa_cum)

        # Plot
        ax.plot(mfp_sorted, kappa_frac, 'b-', lw=2)
        ax.fill_between(mfp_sorted, 0, kappa_frac, alpha=0.2, color='blue')

        # Annotations
        ax.set_xlabel('Mean Free Path (nm)', fontsize=12)
        ax.set_ylabel('Cumulative κ (%)', fontsize=12)
        ax.set_title(f'{formula} — Cumulative Thermal Conductivity',
                     fontsize=13, fontweight='bold')

        # 50% and 90% markers
        for target, color in [(50, 'green'), (90, 'red')]:
            idx = np.argmin(np.abs(kappa_frac - target))
            ax.axvline(x=mfp_sorted[idx], color=color, ls='--', alpha=0.5)
            ax.text(mfp_sorted[idx], target + 2,
                   f'{target}%: {mfp_sorted[idx]:.1f} nm',
                   color=color, fontsize=9, ha='center')

        ax.grid(alpha=0.15)
        return fig, ax

    @staticmethod
    def plot_convergence_mesh(mesh_sizes: List[int],
                             frequencies_at_gamma: List[float],
                             property_name: str = 'Frequency at Γ',
                             ax=None) -> Tuple:
        """Plot convergence of phonon properties vs mesh density.

        Args:
            mesh_sizes: List of mesh sizes (e.g., [5, 8, 10, 12, 15])
            frequencies_at_gamma: Property value at each mesh size
            property_name: Label for the y-axis
            ax: Matplotlib axis

        Returns:
            (fig, ax) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        ax.plot(mesh_sizes, frequencies_at_gamma, 'bo-', lw=2, markersize=6)

        # Convergence threshold
        if len(frequencies_at_gamma) > 1:
            converged_val = frequencies_at_gamma[-1]
            ax.axhline(y=converged_val, color='gray', ls='--', alpha=0.5,
                      label=f'Converged: {converged_val:.2f}')

        ax.set_xlabel('Mesh Size (N×N×N)', fontsize=12)
        ax.set_ylabel(property_name, fontsize=12)
        ax.set_title(f'Convergence vs Mesh Density',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.15)

        return fig, ax

    @staticmethod
    def plot_multi_material_overlay(band_dicts: Dict[str, Dict],
                                   formulas: List[str],
                                   ax=None) -> Tuple:
        """Overlay band structures from multiple materials for comparison.

        Args:
            band_dicts: {formula: band_dict} mapping
            formulas: List of formulas to plot
            ax: Matplotlib axis

        Returns:
            (fig, ax) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            fig = ax.figure

        colors = ['#1E3A8A', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']

        for i, formula in enumerate(formulas):
            if formula not in band_dicts:
                continue
            band_dict = band_dicts[formula]
            color = colors[i % len(colors)]

            distances = band_dict['distances']
            frequencies = band_dict['frequencies']

            for dist, freq in zip(distances, frequencies):
                for b in range(freq.shape[1]):
                    ax.plot(dist, freq[:, b], color=color, lw=0.6, alpha=0.6,
                           label=formula if b == 0 else "")

        ax.legend(fontsize=10)
        ax.set_ylabel('Frequency (THz)', fontsize=12)
        ax.set_title('Multi-Material Band Structure Overlay',
                     fontsize=13, fontweight='bold')
        ax.grid(alpha=0.1)

        return fig, ax

    @staticmethod
    def plot_thermo_comparison(thermo_data: Dict[str, Dict],
                              formulas: List[str],
                              property: str = 'Cv',
                              ax=None) -> Tuple:
        """Compare thermodynamic properties across materials.

        Args:
            thermo_data: {formula: thermo_dict} mapping
            formulas: List of formulas
            property: Property to compare ('Cv', 'S', 'F')
            ax: Matplotlib axis

        Returns:
            (fig, ax) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        colors = ['#1E3A8A', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
        labels = {'Cv': 'C_v (J/mol·K)', 'S': 'S (J/mol·K)', 'F': 'F (kJ/mol)'}
        keys = {'Cv': 'heat_capacity', 'S': 'entropy', 'F': 'free_energy'}

        for i, formula in enumerate(formulas):
            if formula not in thermo_data:
                continue
            data = thermo_data[formula]
            temps = data['temperatures']
            values = data[keys[property]]
            color = colors[i % len(colors)]

            ax.plot(temps, values, color=color, lw=2, label=formula)

        ax.set_xlabel('T (K)', fontsize=12)
        ax.set_ylabel(labels[property], fontsize=12)
        ax.set_title(f'{property} Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.15)

        return fig, ax
