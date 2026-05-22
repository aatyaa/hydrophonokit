"""
=============================================================================
  HydroPhonoKit Postprocessor — Advanced Visualization Engine

  Generates publication-quality plots with configurable themes and
  advanced visualization capabilities.

  Scientific Foundation:
    Phonon visualization follows standard conventions in condensed matter
    physics for band structure and DOS presentation:
      - Band structure: frequency vs q-path with high-symmetry points
      - Fat bands: band width proportional to element projection
      - DOS: total + element-projected with consistent color scheme
      - Thermodynamics: F(T), S(T), C_v(T) with Dulong-Petit limit
      - H-analysis: mode decomposition with labeled peaks
      - Convergence: mesh density, supercell size effects

  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy plotting
    [2] APS Style Guide -- Figure preparation standards
    [3] Nature Publishing Guide -- Color and figure standards
=============================================================================
"""
import os
import warnings
import numpy as np
from typing import Dict, Optional, List, Tuple

# Configure matplotlib backend
import os as _os
if not _os.environ.get('MPLBACKEND'):
    import matplotlib
    matplotlib.use('Agg')
del _os

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# THEME DEFINITIONS
# ============================================================================

THEMES = {
    'nature': {
        'description': 'Nature journal style - clean serif, muted colors',
        'font': {'family': 'serif', 'size': 11, 'weight': 'normal'},
        'figure': {'dpi': 300, 'facecolor': 'white'},
        'axes': {
            'linewidth': 1.0,
            'labelsize': 12,
            'titlesize': 13,
            'grid_alpha': 0.15,
        },
        'colors': {
            'band': '#1E3A8A',
            'band_imaginary': '#EF4444',
            'dos_fill': '#1E3A8A',
            'dos_line': '#1E3A8A',
            'zero_line': '#EF4444',
            'stability_pass': '#059669',
            'stability_fail': '#DC2626',
            'grid': '#94A3B8',
        },
        'legend': {'fontsize': 10, 'framealpha': 0.9},
        'savefig': {'dpi': 300, 'format': 'png'},
    },
    'science': {
        'description': 'Science journal style - bold sans-serif, high contrast',
        'font': {'family': 'sans-serif', 'size': 12, 'weight': 'bold'},
        'figure': {'dpi': 300, 'facecolor': 'white'},
        'axes': {
            'linewidth': 1.5,
            'labelsize': 13,
            'titlesize': 14,
            'grid_alpha': 0.2,
        },
        'colors': {
            'band': '#000000',
            'band_imaginary': '#FF0000',
            'dos_fill': '#000000',
            'dos_line': '#000000',
            'zero_line': '#FF0000',
            'stability_pass': '#00AA00',
            'stability_fail': '#FF0000',
            'grid': '#CCCCCC',
        },
        'legend': {'fontsize': 11, 'framealpha': 0.95},
        'savefig': {'dpi': 300, 'format': 'pdf'},
    },
    'dark': {
        'description': 'Dark theme for presentations and screen viewing',
        'font': {'family': 'sans-serif', 'size': 12, 'weight': 'normal'},
        'figure': {'dpi': 150, 'facecolor': '#1a1a2e'},
        'axes': {
            'linewidth': 1.2,
            'labelsize': 12,
            'titlesize': 13,
            'grid_alpha': 0.1,
        },
        'colors': {
            'band': '#60A5FA',
            'band_imaginary': '#F87171',
            'dos_fill': '#60A5FA',
            'dos_line': '#60A5FA',
            'zero_line': '#F87171',
            'stability_pass': '#34D399',
            'stability_fail': '#F87171',
            'grid': '#333333',
        },
        'legend': {'fontsize': 10, 'framealpha': 0.8},
        'savefig': {'dpi': 150, 'format': 'png',
                    'facecolor': '#1a1a2e', 'edgecolor': '#1a1a2e'},
    },
    'minimal': {
        'description': 'Minimalist style for black-and-white publications',
        'font': {'family': 'serif', 'size': 10, 'weight': 'normal'},
        'figure': {'dpi': 300, 'facecolor': 'white'},
        'axes': {
            'linewidth': 0.8,
            'labelsize': 10,
            'titlesize': 11,
            'grid_alpha': 0.0,
        },
        'colors': {
            'band': '#000000',
            'band_imaginary': '#000000',
            'dos_fill': '#808080',
            'dos_line': '#000000',
            'zero_line': '#000000',
            'stability_pass': '#000000',
            'stability_fail': '#000000',
            'grid': '#FFFFFF',
        },
        'legend': {'fontsize': 9, 'framealpha': 0.0},
        'savefig': {'dpi': 600, 'format': 'eps'},
    },
}


# Extended element color palette
ELEMENT_COLORS = {
    'H': '#F39C12', 'He': '#E67E22',
    'Li': '#E74C3C', 'Be': '#2ECC71', 'B': '#2ECC71', 'C': '#34495E',
    'N': '#9B59B6', 'O': '#E74C3C', 'F': '#1ABC9C', 'Ne': '#85C1E9',
    'Na': '#E74C3C', 'Mg': '#2ECC71', 'Al': '#1ABC9C', 'Si': '#34495E',
    'P': '#9B59B6', 'S': '#F39C12', 'Cl': '#2ECC71', 'Ar': '#85C1E9',
    'K': '#E74C3C', 'Ca': '#3498DB', 'Sc': '#16A085', 'Ti': '#1ABC9C',
    'V': '#16A085', 'Cr': '#138D75', 'Mn': '#7D3C98', 'Fe': '#CB4335',
    'Co': '#8E44AD', 'Ni': '#2980B9', 'Cu': '#D4AC0D', 'Zn': '#7FB3D8',
    'Ga': '#1ABC9C', 'Ge': '#34495E', 'As': '#9B59B6', 'Se': '#F39C12',
    'Br': '#3498DB', 'Kr': '#85C1E9',
    'Rb': '#E74C3C', 'Sr': '#85C1E9', 'Y': '#5DADE2', 'Zr': '#5DADE2',
    'Nb': '#3498DB', 'Mo': '#2874A6', 'Tc': '#1B4F72', 'Ru': '#1A5276',
    'Rh': '#154360', 'Pd': '#1B2631', 'Ag': '#2E4053', 'Cd': '#7FB3D8',
    'In': '#5DADE2', 'Sn': '#34495E', 'Sb': '#9B59B6', 'Te': '#F39C12',
    'I': '#1ABC9C', 'Xe': '#85C1E9',
    'default': '#888888'
}


class PhononPlotter:
    """Advanced publication-quality phonon plotter.

    Supports multiple themes, fat bands, multi-material overlay,
    and various specialized plot types.

    Usage:
        plotter = PhononPlotter(theme='nature')
        plotter.plot_band_structure(band_dict, formula='Si')
        plotter.plot_fat_bands(band_dict, pdos, formula='MgH2')
        plotter.save_all('output_dir/')
    """

    def __init__(self, theme: str = 'nature', interactive: bool = False):
        """
        Args:
            theme: Plot theme name (nature, science, dark, minimal)
            interactive: If True, use interactive backend
        """
        self.theme_name = theme
        self.theme = THEMES.get(theme, THEMES['nature'])
        self.interactive = interactive
        self._figures = []  # Track figures for saving
        self._setup_style()

    def _setup_style(self):
        """Apply theme settings to matplotlib."""
        t = self.theme
        mpl.rcParams.update({
            'font.family': t['font']['family'],
            'font.size': t['font']['size'],
            'font.weight': t['font']['weight'],
            'figure.dpi': t['figure']['dpi'],
            'figure.facecolor': t['figure']['facecolor'],
            'axes.linewidth': t['axes']['linewidth'],
            'axes.labelsize': t['axes']['labelsize'],
            'axes.titlesize': t['axes']['titlesize'],
            'legend.fontsize': t['legend']['fontsize'],
            'legend.framealpha': t['legend']['framealpha'],
            'savefig.dpi': t['savefig']['dpi'],
        })

        if not self.interactive:
            mpl.rcParams['savefig.format'] = t['savefig']['format']

    def _get_color(self, name: str) -> str:
        """Get color from current theme."""
        return self.theme['colors'].get(name, '#888888')

    # ========================================================================
    # BAND STRUCTURE PLOTS
    # ========================================================================

    def plot_band_structure(self, band_dict: Dict, formula: str,
                           ax=None, highlight_imaginary: bool = True,
                           zero_line: bool = True) -> Tuple:
        """Plot phonon band structure.

        Args:
            band_dict: Phonopy band structure dictionary
            formula: Chemical formula for title
            ax: Matplotlib axis (creates new if None)
            highlight_imaginary: Color imaginary modes differently
            zero_line: Draw zero frequency line

        Returns:
            (fig, ax) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        distances = band_dict['distances']
        frequencies = band_dict['frequencies']

        # Check for imaginary modes
        all_f = np.concatenate([f.flatten() for f in frequencies])
        has_imaginary = np.any(all_f < -0.1)

        # Plot bands
        for dist, freq in zip(distances, frequencies):
            for b in range(freq.shape[1]):
                if highlight_imaginary and has_imaginary:
                    # Color bands based on imaginary/real
                    for i in range(len(dist) - 1):
                        f_avg = (freq[i, b] + freq[i+1, b]) / 2
                        color = self._get_color('band_imaginary') if f_avg < -0.1 else self._get_color('band')
                        ax.plot(dist[i:i+2], freq[i:i+2, b], color=color, lw=0.8, alpha=0.85)
                else:
                    ax.plot(dist, freq[:, b], color=self._get_color('band'), lw=0.8, alpha=0.85)

        # High-symmetry point markers
        sp = [distances[0][0]]
        for d in distances:
            sp.append(d[-1])
        for xp in sp:
            ax.axvline(x=xp, color=self._get_color('grid'), lw=0.5)

        # Zero frequency line
        if zero_line:
            ax.axhline(y=0, color=self._get_color('zero_line'), lw=1.0, ls='--', alpha=0.7)

        # Stability badge
        mf = np.min(all_f)
        if mf < -0.5:
            badge_text = f'⚠ Imaginary: {mf:.2f} THz'
            badge_color = self._get_color('stability_fail')
            badge_bg = '#FEE2E2' if self.theme_name != 'dark' else '#3D0000'
        else:
            badge_text = '✓ Dynamically Stable'
            badge_color = self._get_color('stability_pass')
            badge_bg = '#D1FAE5' if self.theme_name != 'dark' else '#003D1D'

        ax.text(0.02, 0.02, badge_text,
                transform=ax.transAxes, color=badge_color, fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=badge_bg, alpha=0.9))

        ax.set_ylabel('Frequency (THz)', fontsize=self.theme['axes']['labelsize'])
        ax.set_title(f'{formula} — Phonon Band Structure',
                     fontsize=self.theme['axes']['titlesize'], fontweight='bold')
        ax.set_xlim(distances[0][0], distances[-1][-1])

        self._figures.append(fig)
        return fig, ax

    def plot_fat_bands(self, band_dict: Dict, pdos_data: Tuple,
                      formula: str, element: str = None,
                      ax=None, max_width: float = 3.0) -> Tuple:
        """Plot element-projected ('fat') band structure.

        Band width is proportional to the element's contribution to that mode.

        Args:
            band_dict: Phonopy band structure dictionary
            pdos_data: (freq, pdos, sym_to_idx) tuple
            formula: Chemical formula
            element: Specific element to project (None = all)
            ax: Matplotlib axis
            max_width: Maximum band width in points

        Returns:
            (fig, ax) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        freq, pdos_arr, sym_to_idx = pdos_data

        # Compute mode projections for each band
        # This requires eigenvector data which we approximate from pDOS
        elements = [element] if element else list(sym_to_idx.keys())

        for dist, freq_arr in zip(distances, frequencies):
            for b in range(freq_arr.shape[1]):
                f_val = freq_arr[0, b]  # Approximate frequency
                if f_val < 0:
                    continue

                # Find DOS contribution at this frequency
                freq_idx = np.argmin(np.abs(freq - abs(f_val)))
                total_dos = pdos_arr[:, freq_idx].sum()

                if total_dos < 1e-10:
                    continue

                for elem in elements:
                    if elem not in sym_to_idx:
                        continue

                    indices = sym_to_idx[elem]
                    elem_dos = sum(pdos_arr[i, freq_idx] for i in indices if i < pdos_arr.shape[0])
                    fraction = elem_dos / total_dos

                    # Band width proportional to contribution
                    width = max_width * fraction
                    color = ELEMENT_COLORS.get(elem, ELEMENT_COLORS['default'])

                    for i in range(len(dist) - 1):
                        ax.plot(dist[i:i+2], freq_arr[i:i+2, b],
                               color=color, lw=width, alpha=0.7)

        # Standard band structure overlay (thin lines)
        for dist, freq_arr in zip(distances, frequencies):
            for b in range(freq_arr.shape[1]):
                ax.plot(dist, freq_arr[:, b], color='black', lw=0.5, alpha=0.3, zorder=0)

        # High-symmetry markers
        sp = [distances[0][0]]
        for d in distances:
            sp.append(d[-1])
        for xp in sp:
            ax.axvline(x=xp, color=self._get_color('grid'), lw=0.5)

        ax.axhline(y=0, color=self._get_color('zero_line'), lw=1.0, ls='--', alpha=0.7)

        # Legend
        legend_elements = [plt.Line2D([0], [0], color=ELEMENT_COLORS.get(e, '#888'),
                                       lw=3, label=e) for e in elements]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        ax.set_ylabel('Frequency (THz)', fontsize=self.theme['axes']['labelsize'])
        ax.set_title(f'{formula} — Fat Band Structure',
                     fontsize=self.theme['axes']['titlesize'], fontweight='bold')
        ax.set_xlim(distances[0][0], distances[-1][-1])

        self._figures.append(fig)
        return fig, ax

    # ========================================================================
    # DOS PLOTS
    # ========================================================================

    def plot_partial_dos(self, pdos_data: Tuple, formula: str,
                        ax_top=None, ax_bottom=None,
                        group_by: str = None) -> Tuple:
        """Plot partial DOS with optional element grouping.

        Args:
            pdos_data: (freq, pdos, sym_to_idx) tuple
            formula: Chemical formula
            ax_top: Top axis for total DOS
            ax_bottom: Bottom axis for partial DOS
            group_by: None, 'metal/nonmetal', or custom dict

        Returns:
            (fig, (ax_top, ax_bottom)) tuple
        """
        freq, pdos_arr, sym_to_idx = pdos_data
        total_dos = pdos_arr.sum(axis=0)

        if ax_top is None or ax_bottom is None:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 7), sharex=True,
                gridspec_kw={'height_ratios': [1, 2]}
            )
        else:
            ax1, ax2 = ax_top, ax_bottom
            fig = ax1.figure

        # Total DOS
        ax1.fill_between(freq, total_dos, alpha=0.3, color=self._get_color('dos_fill'))
        ax1.plot(freq, total_dos, color=self._get_color('dos_line'), lw=1.2, label='Total')
        ax1.set_ylabel('DOS (states/THz)', fontsize=self.theme['axes']['labelsize'])
        ax1.set_title(f'{formula} — Phonon Density of States',
                      fontsize=self.theme['axes']['titlesize'], fontweight='bold')
        ax1.legend(fontsize=self.theme['legend']['fontsize'])

        pos_mask = total_dos > 0.001
        if np.any(pos_mask):
            ax1.set_xlim(0, max(freq[pos_mask]) * 1.05)

        # Partial DOS by element or group
        if group_by == 'metal/nonmetal':
            groups = self._group_elements_metal_nonmetal(sym_to_idx)
        else:
            groups = {elem: idxs for elem, idxs in sym_to_idx.items()}

        for group_name, indices in groups.items():
            sp_dos = np.zeros_like(freq)
            for idx in indices:
                if idx < pdos_arr.shape[0]:
                    sp_dos += pdos_arr[idx]

            c = ELEMENT_COLORS.get(group_name, ELEMENT_COLORS['default'])
            ax2.fill_between(freq, sp_dos, alpha=0.2, color=c)
            ax2.plot(freq, sp_dos, color=c, lw=1.5, label=group_name)

        ax2.set_xlabel('Frequency (THz)', fontsize=self.theme['axes']['labelsize'])
        ax2.set_ylabel('Partial DOS', fontsize=self.theme['axes']['labelsize'])
        ax2.legend(fontsize=10, ncol=4, loc='upper right')
        ax2.grid(alpha=self.theme['axes']['grid_alpha'])

        self._figures.append(fig)
        return fig, (ax1, ax2)

    def _group_elements_metal_nonmetal(self, sym_to_idx: Dict) -> Dict:
        """Group elements into metals and non-metals."""
        metals = set()
        nonmetals = set()

        for elem in sym_to_idx.keys():
            if elem in {'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                        'Si', 'P', 'S', 'Cl', 'Ar', 'Se', 'Br', 'Kr',
                        'I', 'Xe'}:
                nonmetals.add(elem)
            else:
                metals.add(elem)

        groups = {}
        if metals:
            groups['Metals'] = [i for elem in metals for i in sym_to_idx.get(elem, [])]
        if nonmetals:
            groups['Non-metals'] = [i for elem in nonmetals for i in sym_to_idx.get(elem, [])]
        return groups

    # ========================================================================
    # THERMODYNAMICS PLOTS
    # ========================================================================

    def plot_thermodynamics(self, thermo_data: Dict, formula: str,
                           n_atoms: int, show_dulong_petit: bool = True,
                           fig=None) -> Tuple:
        """Plot F(T), S(T), C_v(T) with optional Dulong-Petit limit."""
        temps = thermo_data['temperatures']
        fe = thermo_data['free_energy']
        ent = thermo_data['entropy']
        cv = thermo_data['heat_capacity']

        dp_limit = 3 * n_atoms * 8.31446

        if fig is None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
        else:
            axes = fig.get_axes()
            ax1, ax2, ax3 = axes[0], axes[1], axes[2]

        # Free Energy
        ax1.plot(temps, fe, color=self._get_color('band'), lw=2)
        ax1.set_xlabel('T (K)')
        ax1.set_ylabel('F (kJ/mol)')
        ax1.set_title('F(T)', fontweight='bold')
        ax1.grid(alpha=self.theme['axes']['grid_alpha'])

        # Entropy
        ax2.plot(temps, ent, color='#E74C3C', lw=2)
        ax2.set_xlabel('T (K)')
        ax2.set_ylabel('S (J/mol·K)')
        ax2.set_title('S(T)', fontweight='bold')
        ax2.grid(alpha=self.theme['axes']['grid_alpha'])

        # Heat Capacity
        ax3.plot(temps, cv, color='#2ECC71', lw=2)
        if show_dulong_petit:
            ax3.axhline(y=dp_limit, color='gray', ls='--', alpha=0.5,
                        label=f'Dulong-Petit = {dp_limit:.0f}')
        ax3.set_xlabel('T (K)')
        ax3.set_ylabel('Cv (J/mol·K)')
        ax3.set_title('Cv(T)', fontweight='bold')
        if show_dulong_petit:
            ax3.legend(fontsize=9)
        ax3.grid(alpha=self.theme['axes']['grid_alpha'])

        fig.suptitle(f'{formula} — Vibrational Thermodynamics',
                     fontsize=15, fontweight='bold', y=1.02)
        fig.tight_layout()

        self._figures.append(fig)
        return fig, (ax1, ax2, ax3)

    # ========================================================================
    # HYDROGEN MODE PLOTS
    # ========================================================================

    def plot_h_modes(self, pdos_data, h_metrics: Dict, formula: str,
                    ax=None) -> Tuple:
        """Plot hydrogen vibrational mode decomposition."""
        freq, pdos_arr, sym_to_idx = pdos_data

        h_dos = np.zeros_like(freq)
        for idx in sym_to_idx.get('H', []):
            if idx < pdos_arr.shape[0]:
                h_dos += pdos_arr[idx]

        lib_mask = (freq > 5) & (freq < 20.9)
        bend_mask = (freq > 20.9) & (freq < 50)
        stretch_mask = (freq > 50) & (freq < 100)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.fill_between(freq[lib_mask], h_dos[lib_mask],
                       alpha=0.3, color='#9B59B6', label='Librational')
        ax.fill_between(freq[bend_mask], h_dos[bend_mask],
                       alpha=0.3, color='#3498DB', label='Bending')
        ax.fill_between(freq[stretch_mask], h_dos[stretch_mask],
                       alpha=0.3, color='#E74C3C', label='H Stretching')
        ax.plot(freq, h_dos, color='#F39C12', lw=1.5, label='H total')

        if 'peak_thz' in h_metrics:
            ax.axvline(x=h_metrics['peak_thz'], color='#E74C3C', ls='--', alpha=0.7)
            ax.text(h_metrics['peak_thz'] + 1, max(h_dos) * 0.85,
                    f"Peak\n{h_metrics['peak_thz']:.1f} THz\n({h_metrics['peak_cm']:.0f} cm⁻¹)",
                    fontsize=9, color='#E74C3C', fontweight='bold')

        ax.set_xlabel('Frequency (THz)', fontsize=self.theme['axes']['labelsize'])
        ax.set_ylabel('H Partial DOS', fontsize=self.theme['axes']['labelsize'])
        ax.set_title(f'{formula} — Hydrogen Mode Decomposition',
                     fontsize=self.theme['axes']['titlesize'], fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=self.theme['axes']['grid_alpha'])

        pos_mask = h_dos > 0.001
        if np.any(pos_mask):
            ax.set_xlim(0, max(freq[pos_mask]) * 1.05)

        self._figures.append(fig)
        return fig, ax

    # ========================================================================
    # SAVING
    # ========================================================================

    def save(self, path: str, fig=None, **kwargs):
        """Save the last or specified figure.

        Args:
            path: Output path (extension determines format)
            fig: Figure to save (saves last if None)
            **kwargs: Additional savefig arguments
        """
        if fig is None:
            if not self._figures:
                raise ValueError("No figures to save.")
            fig = self._figures[-1]

        save_kwargs = {**self.theme['savefig'], **kwargs}
        fig.savefig(path, **save_kwargs)

    def save_all(self, output_dir: str, format: str = None):
        """Save all tracked figures.

        Args:
            output_dir: Directory to save figures
            format: Override format ('png', 'pdf', 'eps', etc.)
        """
        os.makedirs(output_dir, exist_ok=True)

        base_names = [
            'phonon_band_structure',
            'phonon_dos_partial',
            'phonon_thermodynamics',
            'H_mode_analysis',
        ]

        for i, fig in enumerate(self._figures):
            name = base_names[i] if i < len(base_names) else f'plot_{i}'
            ext = format or self.theme['savefig']['format']
            path = os.path.join(output_dir, f'{name}.{ext}')
            fig.savefig(path, **self.theme['savefig'])
            print(f"  --> Saved: {path}")

    def close_all(self):
        """Close all tracked figures."""
        for fig in self._figures:
            plt.close(fig)
        self._figures = []
