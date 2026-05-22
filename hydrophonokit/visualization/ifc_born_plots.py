"""
=============================================================================
  HydroPhonoKit Visualization — Force Constants & Born Charges Plots

  Implements force constants and Born effective charge visualization:
    1. IFC Decay Plot - |Φ(r)| vs interatomic distance
    2. IFC Heatmap - Force constant matrix visualization
    3. Born Effective Charges - Z* tensor per atom
    4. Dielectric Tensor - ε tensor eigenvalues
    5. Charge Neutrality Check - Σ Z* = 0 validation
    6. Range of Interactions - Cutoff distance analysis

  Scientific Foundation:
    Interatomic force constants (IFCs) describe the harmonic potential:
      V = V0 + 1/2 Σ Φ_αβ(lκ, l'κ') u_α(lκ) u_β(l'κ')

    Born effective charges Z* describe the change in polarization:
      Z*_{κ,αβ} = (Ω/e) ∂P_α/∂u_{κ,β}

    Key validations:
      - Acoustic Sum Rule: Σ Φ → 0 at long range
      - Charge Neutrality: Σ Z* = 0
      - Symmetry: Φ_αβ = Φ_βα (Hermitian)
      - Range: IFCs should decay to zero within supercell

  References:
    [1] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT
    [2] Gonze & Lee, Phys. Rev. B 55, 10355 (1997) -- Born charges
    [3] Wang et al., Phys. Rev. B 95, 014303 (2017) -- symfc
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
from matplotlib.colors import LogNorm

from .base_plotter import BasePlotter


# ============================================================================
# IFC & BORN PLOTTER CLASS
# ============================================================================

class IFCBornPlotter(BasePlotter):
    """Specialized plotter for force constants and Born charges.
    
    Provides methods for IFC and Born charge visualization.
    """
    
    def plot_ifc_decay(self, distances: np.ndarray,
                      ifc_magnitudes: np.ndarray,
                      formula: str = 'Material',
                      ax: mpl_axes.Axes = None,
                      figsize: Tuple[float, float] = (8, 6),
                      title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot IFC magnitude vs interatomic distance.
        
        Shows how force constants decay with distance, validating
        that the supercell is large enough for convergence.
        
        Args:
            distances: Interatomic distances (Å)
            ifc_magnitudes: |Φ_αβ| values (eV/Å²)
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
        
        # Scatter plot with log y-axis
        ax.semilogy(distances, np.maximum(ifc_magnitudes, 1e-10),
                   'o', color=self.get_color('primary'), markersize=4, alpha=0.7)
        
        # Fitted decay envelope
        if len(distances) > 2:
            # Fit exponential decay: |Φ| = A * exp(-r/ξ)
            mask = ifc_magnitudes > 1e-8
            if np.sum(mask) > 2:
                from scipy import optimize
                def exp_decay(r, A, xi):
                    return A * np.exp(-r / xi)
                
                try:
                    popt, _ = optimize.curve_fit(
                        exp_decay, distances[mask], ifc_magnitudes[mask],
                        p0=[np.max(ifc_magnitudes[mask]), np.max(distances[mask])/3]
                    )
                    r_fit = np.linspace(0, max(distances), 100)
                    ax.semilogy(r_fit, exp_decay(r_fit, *popt),
                               color=self.get_color('secondary'), lw=2,
                               label=f'ξ = {popt[1]:.2f} Å')
                    ax.legend()
                except Exception:
                    pass
        
        ax.set_xlabel('Interatomic Distance (Å)')
        ax.set_ylabel('|Φ| (eV/Å²)')
        ax.grid(alpha=self.theme['axes']['grid_alpha'], which='both')
        
        # Annotation
        if len(ifc_magnitudes) > 0:
            max_ifc = np.max(ifc_magnitudes)
            min_ifc = np.min(ifc_magnitudes[ifc_magnitudes > 0])
            ratio = max_ifc / min_ifc if min_ifc > 0 else np.inf
            ax.text(0.02, 0.95,
                   f'Max: {max_ifc:.2f}\nMin: {min_ifc:.2e}\nRatio: {ratio:.0e}',
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if title is None:
            title = f'{formula} - IFC Decay vs Distance'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_ifc_heatmap(self, ifc_matrix: np.ndarray,
                        formula: str = 'Material',
                        ax: mpl_axes.Axes = None,
                        figsize: Tuple[float, float] = (8, 6),
                        title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot force constant matrix as heatmap.
        
        Shows the full IFC matrix structure, revealing:
          - Block diagonal structure (atom pairs)
          - Symmetry patterns
          - Magnitude variations
        
        Args:
            ifc_matrix: IFC matrix (3N × 3N)
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
        
        n = ifc_matrix.shape[0]
        max_val = np.max(np.abs(ifc_matrix))
        
        # Heatmap with diverging colormap
        im = ax.imshow(ifc_matrix, cmap='RdBu_r',
                      norm=plt.Normalize(-max_val, max_val),
                      aspect='equal')
        
        fig.colorbar(im, ax=ax, label='Φ (eV/Å²)')
        
        # Grid lines at atom boundaries
        n_atoms = n // 3
        if n_atoms > 1:
            for i in range(1, n_atoms):
                ax.axhline(y=i*3 - 0.5, color='gray', lw=0.5, alpha=0.5)
                ax.axvline(x=i*3 - 0.5, color='gray', lw=0.5, alpha=0.5)
        
        ax.set_xlabel('Cartesian Index')
        ax.set_ylabel('Cartesian Index')
        
        if title is None:
            title = f'{formula} - IFC Heatmap ({n_atoms} atoms)'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_born_charges(self, born_charges: np.ndarray,
                         formula: str = 'Material',
                         ax: mpl_axes.Axes = None,
                         figsize: Tuple[float, float] = (8, 6),
                         title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot Born effective charge tensors.
        
        Shows Z* tensor for each atom, revealing:
          - Charge transfer
          - Anisotropy
          - Deviation from nominal charges
        
        Args:
            born_charges: Born charges (n_atoms × 3 × 3)
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
        
        n_atoms = born_charges.shape[0]
        
        # Extract tensor traces (effective charges)
        traces = np.array([np.trace(born_charges[i]) for i in range(n_atoms)])
        x = np.arange(n_atoms)
        
        # Bar plot
        colors = [self.get_color('primary') if t > 0 else self.get_color('secondary')
                 for t in traces]
        ax.bar(x, traces, color=colors, alpha=0.7, edgecolor='black')
        
        # Zero line
        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
        
        # Sum annotation
        total_charge = np.sum(traces)
        ax.text(0.02, 0.95, f'Σ Z* = {total_charge:.3f}',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Atom Index')
        ax.set_ylabel('Tr(Z*) (effective charge)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in range(n_atoms)])
        ax.grid(alpha=self.theme['axes']['grid_alpha'], axis='y')
        
        if title is None:
            title = f'{formula} - Born Effective Charges'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_dielectric_tensor(self, dielectric: np.ndarray,
                              formula: str = 'Material',
                              ax: mpl_axes.Axes = None,
                              figsize: Tuple[float, float] = (6, 6),
                              title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot dielectric tensor as heatmap with eigenvalues.
        
        Shows:
          - Full ε tensor
          - Eigenvalues (principal dielectric constants)
          - Average dielectric constant
        
        Args:
            dielectric: Dielectric tensor (3 × 3)
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
        
        # Heatmap
        max_val = np.max(np.abs(dielectric))
        im = ax.imshow(dielectric, cmap='Blues',
                      norm=plt.Normalize(0, max_val),
                      aspect='equal')
        
        fig.colorbar(im, ax=ax, label='ε')
        
        # Value annotations
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{dielectric[i,j]:.2f}',
                       ha='center', va='center', fontsize=10,
                       color='white' if dielectric[i,j] > max_val*0.6 else 'black')
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(dielectric)
        ax.text(0.02, 0.02,
               f'Eigenvalues:\n'
               f'  ε₁ = {eigenvalues[0]:.2f}\n'
               f'  ε₂ = {eigenvalues[1]:.2f}\n'
               f'  ε₃ = {eigenvalues[2]:.2f}\n'
               f'  Avg = {np.mean(eigenvalues):.2f}',
               transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Component')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['x', 'y', 'z'])
        ax.set_yticklabels(['x', 'y', 'z'])
        
        if title is None:
            title = f'{formula} - Dielectric Tensor'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_interaction_range(self, distances: np.ndarray,
                              ifc_magnitudes: np.ndarray,
                              formula: str = 'Material',
                              cutoff_threshold: float = 1e-3,
                              ax: mpl_axes.Axes = None,
                              figsize: Tuple[float, float] = (8, 6),
                              title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot interaction range analysis.
        
        Determines the cutoff distance where IFCs become negligible.
        
        Args:
            distances: Interatomic distances (Å)
            ifc_magnitudes: |Φ_αβ| values (eV/Å²)
            formula: Material formula
            cutoff_threshold: Threshold for "negligible" IFC
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
        
        # Main plot (linear)
        ax.semilogy(distances, np.maximum(ifc_magnitudes, 1e-12),
                   'o-', color=self.get_color('primary'), markersize=4, lw=1)
        
        # Cutoff line
        cutoff_idx = np.where(ifc_magnitudes < cutoff_threshold)[0]
        if len(cutoff_idx) > 0:
            cutoff_dist = distances[cutoff_idx[0]]
            ax.axvline(x=cutoff_dist, color=self.get_color('secondary'),
                      ls='--', alpha=0.7,
                      label=f'Cutoff ({cutoff_threshold:.0e}): {cutoff_dist:.2f} Å')
            ax.axhline(y=cutoff_threshold, color=self.get_color('secondary'),
                      ls=':', alpha=0.5)
            ax.legend()
        
        # Statistics
        if len(ifc_magnitudes) > 0:
            n_above = np.sum(ifc_magnitudes >= cutoff_threshold)
            n_total = len(ifc_magnitudes)
            ax.text(0.02, 0.95,
                   f'{n_above}/{n_total} interactions above threshold',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Interatomic Distance (Å)')
        ax.set_ylabel('|Φ| (eV/Å²)')
        ax.grid(alpha=self.theme['axes']['grid_alpha'], which='both')
        
        if title is None:
            title = f'{formula} - Interaction Range Analysis'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
    
    def plot_charge_neutrality(self, born_charges: np.ndarray,
                              formula: str = 'Material',
                              ax: mpl_axes.Axes = None,
                              figsize: Tuple[float, float] = (6, 4),
                              title: str = None) -> Tuple[Figure, mpl_axes.Axes]:
        """Plot charge neutrality check: Σ Z* = 0.
        
        Validates the Acoustic Sum Rule for Born charges.
        
        Args:
            born_charges: Born charges (n_atoms × 3 × 3)
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
        
        # Sum over atoms
        z_sum = np.sum(born_charges, axis=0)  # 3 × 3
        z_trace = np.trace(z_sum)
        
        # Bar plot of sum components
        components = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        values = z_sum.flatten()
        x = np.arange(9)
        
        colors = [self.get_color('primary') if abs(v) < 0.01
                 else self.get_color('secondary') for v in values]
        ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45)
        ax.set_ylabel('Σ Z*_{αβ}')
        ax.grid(alpha=self.theme['axes']['grid_alpha'], axis='y')
        
        # Annotation
        ax.text(0.02, 0.95,
               f'Tr(Σ Z*) = {z_trace:.6f}\n'
               f'ASR Error: {abs(z_trace):.2e}',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if title is None:
            title = f'{formula} - Charge Neutrality Check'
        ax.set_title(title, fontweight='bold')
        
        fig.tight_layout()
        self._figures.append(fig)
        
        return fig, ax
