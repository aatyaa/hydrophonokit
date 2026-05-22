"""
=============================================================================
  HydroPhonoKit Visualization — Figure Factory

  High-level factory pattern for generating all phonon plot types:
    - Band structure plots
    - DOS plots
    - Thermodynamic plots
    - And more (extensible for future plot types)

  Usage:
      factory = PhononFigureFactory(results, theme='nature')
      factory.plot_band_structure()
      factory.plot_dos()
      factory.generate_all('output_dir/')
=============================================================================
"""
import os
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from .base_plotter import BasePlotter, freq_to_cm, freq_to_meV
from .themes import ELEMENT_COLORS, DEFAULT_THEME
from .band_plots import BandStructurePlotter
from .dos_plots import DOSPlotter
from .thermo_plots import ThermoPlotter
from .ifc_born_plots import IFCBornPlotter
from .transport_plots import TransportPlotter
from .hydrogen_plots import HydrogenPlotter
from .comparison_plots import ComparisonPlotter
from .interactive_plots import InteractivePlotter
from .publication_formatter import PublicationFormatter
from .multi_panel import MultiPanelComposer


class PhononFigureFactory:
    """Factory for generating all phonon figure types.
    
    Provides a unified API for creating publication-ready figures
    from phonon calculation results.
    
    Usage:
        factory = PhononFigureFactory(results, theme='nature')
        
        # Generate specific plots
        factory.plot_band_structure(save=True, output_dir='figures/')
        factory.plot_dos(save=True, output_dir='figures/')
        factory.plot_thermodynamics(save=True, output_dir='figures/')
        
        # Or generate all at once
        factory.generate_all(output_dir='figures/')
    """
    
    def __init__(self, results: Dict, theme: str = DEFAULT_THEME):
        """
        Args:
            results: Postprocessor results dictionary
            theme: Theme name ('nature', 'science', 'prl', 'acs', 'presentation', 'minimal')
        """
        self.results = results
        self.theme_name = theme
        self.band_plotter = BandStructurePlotter(theme=theme)
        self.dos_plotter = DOSPlotter(theme=theme)
        self.thermo_plotter = ThermoPlotter(theme=theme)
        self.ifc_born_plotter = IFCBornPlotter(theme=theme)
        self.transport_plotter = TransportPlotter(theme=theme)
        self.hydrogen_plotter = HydrogenPlotter(theme=theme)
        self.comparison_plotter = ComparisonPlotter(theme=theme)
        self.interactive_plotter = InteractivePlotter(theme=theme)
        self.pub_formatter = PublicationFormatter(theme=theme)
        self.multi_panel = MultiPanelComposer(theme=theme)
        # Backward compatibility alias
        self.plotter = self.pub_formatter
        
        # Extract commonly used data
        self._extract_data()
    
    def _get_n_atoms(self) -> int:
        """Get number of atoms from phonon or profile."""
        if self.phonon:
            return len(self.phonon.unitcell.numbers)
        return getattr(self.profile, 'n_atoms', 1)
    
    def _get_thermo_data(self):
        """Extract thermodynamic data from results."""
        thermo = self.thermo or {}
        thermo_data = thermo.get('thermo_data') or {}
        return {
            'temps': thermo_data.get('temperatures', np.array([])),
            'free_energy': thermo_data.get('free_energy', np.array([])),
            'entropy': thermo_data.get('entropy', np.array([])),
            'cv': thermo_data.get('heat_capacity', np.array([])),
            'zpe': thermo.get('zpe', 0),
            'validations': thermo.get('validations', {}),
            'at_300K': thermo.get('at_300K', {}),
        }
    
    def _extract_data(self):
        """Extract commonly used data from results dictionary."""
        # Data loader
        dl = self.results.get('data_loader') or {}
        self.phonon = dl.get('phonon')
        self.profile = self.results.get('profile')
        
        # Phase results
        self.bands = self.results.get('bands_dos') or {}
        self.thermo = self.results.get('thermodynamics') or {}
        self.hydrogen = self.results.get('hydrogen') or {}
        self.fc = self.results.get('ifc') or {}
        self.gv = self.results.get('group_velocities') or {}
        self.dw = self.results.get('debye_waller') or {}
        self.mrt = self.results.get('mode_resolved_thermo') or {}
        
        # Formula
        self.formula = getattr(self.profile, 'formula', 'Unknown') if self.profile else 'Unknown'
    
    # ========================================================================
    # BAND STRUCTURE PLOTS
    # ========================================================================
    
    def plot_band_structure(self, save: bool = False, output_dir: str = None,
                           unit: str = 'THz', highlight_imaginary: bool = True,
                           **kwargs) -> Tuple:
        """Plot phonon band structure.
        
        Args:
            save: Save figure to file
            output_dir: Output directory (required if save=True)
            unit: Frequency unit ('THz', 'cm^-1', 'meV')
            highlight_imaginary: Color imaginary modes differently
            **kwargs: Additional plot arguments
        
        Returns:
            (fig, ax) tuple
        """
        if not self.bands:
            raise ValueError("No band structure data available")
        
        band_dict = self.bands.get('band_dict')
        if not band_dict:
            raise ValueError("No band_dict in bands_dos results")
        
        fig, ax = self.band_plotter.plot_standard(
            band_dict,
            formula=self.formula,
            unit=kwargs.get('unit', 'THz'),
            highlight_imaginary=kwargs.get('highlight_imaginary', True),
            title=kwargs.get('title', None),
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'phonon_band_structure.png')
            self.band_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_fat_bands(self, save: bool = False, output_dir: str = None,
                      elements: List[str] = None, max_width: float = 3.0,
                      **kwargs) -> Tuple:
        """Plot element-projected ('fat') band structure.
        
        Args:
            save: Save figure to file
            output_dir: Output directory
            elements: List of elements to show (None = all)
            max_width: Maximum band width in points
            **kwargs: Additional plot arguments
        
        Returns:
            (fig, ax) tuple
        """
        if not self.bands:
            raise ValueError("No band structure data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available for fat bands")
        
        import matplotlib.pyplot as plt
        fig, ax = self.plotter.create_figure(figsize=(10, 7))
        
        band_dict = self.bands['band_dict']
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']
        freq, pdos_arr, sym_to_idx = pdos_data
        
        elements_to_show = elements or list(sym_to_idx.keys())
        
        for dist, freq_arr in zip(distances, frequencies):
            for b in range(freq_arr.shape[1]):
                f_val = freq_arr[0, b]
                if f_val < 0:
                    continue
                
                # Find DOS contribution at this frequency
                freq_idx = np.argmin(np.abs(freq - abs(f_val)))
                total_dos = pdos_arr[:, freq_idx].sum()
                
                if total_dos < 1e-10:
                    continue
                
                for elem in elements_to_show:
                    if elem not in sym_to_idx:
                        continue
                    
                    indices = sym_to_idx[elem]
                    elem_dos = sum(pdos_arr[i, freq_idx] for i in indices if i < pdos_arr.shape[0])
                    fraction = elem_dos / total_dos
                    
                    width = max_width * fraction
                    color = self.plotter.get_element_color(elem)
                    
                    for i in range(len(dist) - 1):
                        ax.plot(dist[i:i+2], freq_arr[i:i+2, b],
                               color=color, lw=width, alpha=0.7)
        
        # Background bands
        for dist, freq_arr in zip(distances, frequencies):
            for b in range(freq_arr.shape[1]):
                ax.plot(dist, freq_arr[:, b], color='black', lw=0.5, alpha=0.3, zorder=0)
        
        # Markers and zero line
        sp = [distances[0][0]]
        for d in distances:
            sp.append(d[-1])
        self.plotter.add_high_symmetry_markers(ax, sp)
        self.plotter.add_zero_line(ax)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.plotter.get_element_color(e), lw=3, label=e)
            for e in elements_to_show
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        self.plotter.format_frequency_axis(ax)
        ax.set_title(f'{self.formula} - Fat Band Structure', fontweight='bold')
        ax.set_xlim(distances[0][0], distances[-1][-1])
        
        fig.tight_layout()
        
        if save and output_dir:
            path = os.path.join(output_dir, 'phonon_fat_bands.png')
            self.band_plotter.save(fig, path)
        
        return fig, ax
    
    # ========================================================================
    # ADVANCED BAND STRUCTURE PLOTS
    # ========================================================================
    
    def plot_orbital_projected_bands(self, mode_characters: Dict = None,
                                    save: bool = False, output_dir: str = None,
                                    **kwargs) -> Tuple:
        """Plot mode-character projected bands."""
        if not self.bands:
            raise ValueError("No band structure data available")
        
        band_dict = self.bands['band_dict']
        if mode_characters is None:
            mode_characters = {i: 'optical' for i in range(100)}
            for i in range(3):
                mode_characters[i] = 'acoustic'
        
        fig, ax = self.band_plotter.plot_orbital_projected(
            band_dict, mode_characters, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'orbital_projected_bands.png')
            self.band_plotter.save(fig, path)
        return fig, ax
    
    def plot_imaginary_mode_analysis(self, save: bool = False,
                                    output_dir: str = None, **kwargs) -> Tuple:
        """Plot detailed imaginary mode analysis."""
        if not self.bands:
            raise ValueError("No band structure data available")
        
        fig, ax = self.band_plotter.plot_imaginary_modes(
            self.bands['band_dict'], formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'imaginary_mode_analysis.png')
            self.band_plotter.save(fig, path)
        return fig, ax
    
    def plot_zoomed_band_structure(self, freq_range: Tuple[float, float] = (0, 5),
                                  save: bool = False, output_dir: str = None,
                                  **kwargs) -> Tuple:
        """Plot zoomed-in view of specific frequency range."""
        if not self.bands:
            raise ValueError("No band structure data available")
        
        fig, ax = self.band_plotter.plot_zoomed(
            self.bands['band_dict'], formula=self.formula, freq_range=freq_range, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, f'band_zoomed_{int(freq_range[0])}_{int(freq_range[1])}.png')
            self.band_plotter.save(fig, path)
        return fig, ax
    
    def plot_band_with_experiment(self, exp_data: Dict, save: bool = False,
                                 output_dir: str = None, **kwargs) -> Tuple:
        """Plot band structure overlaid with experimental data."""
        if not self.bands:
            raise ValueError("No band structure data available")
        
        fig, ax = self.band_plotter.plot_with_experiment(
            self.bands['band_dict'], exp_data, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'band_with_experiment.png')
            self.band_plotter.save(fig, path)
        return fig, ax
    
    def plot_all_band_types(self, output_dir: str, mode_characters: Dict = None,
                           exp_data: Dict = None):
        """Generate all band structure plot types."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"  [Bands] Generating all band structure plots in {output_dir}")
        
        # 1. Standard
        try:
            self.plot_band_structure(save=True, output_dir=output_dir)
            print(f"    [OK] Standard band structure")
        except Exception as e:
            print(f"    [FAIL] Standard band structure: {e}")
        
        # 2. Fat bands
        if self.bands and self.bands.get('pdos_data'):
            try:
                self.plot_fat_bands(save=True, output_dir=output_dir)
                print(f"    [OK] Fat bands")
            except Exception as e:
                print(f"    [FAIL] Fat bands: {e}")
        
        # 3. Orbital-projected
        if mode_characters:
            try:
                self.plot_orbital_projected_bands(mode_characters, save=True, output_dir=output_dir)
                print(f"    [OK] Orbital-projected bands")
            except Exception as e:
                print(f"    [FAIL] Orbital-projected bands: {e}")
        
        # 4. Imaginary mode analysis
        if self.bands:
            band_dict = self.bands.get('band_dict', {})
            all_f = np.concatenate([f.flatten() for f in band_dict.get('frequencies', [])])
            if np.any(all_f < -0.1):
                try:
                    self.plot_imaginary_mode_analysis(save=True, output_dir=output_dir)
                    print(f"    [OK] Imaginary mode analysis")
                except Exception as e:
                    print(f"    [FAIL] Imaginary mode analysis: {e}")
        
        # 5. Zoomed regions
        for f_min, f_max in [(0, 5), (5, 20), (20, 50)]:
            try:
                self.plot_zoomed_band_structure((f_min, f_max), save=True, output_dir=output_dir)
                print(f"    [OK] Zoomed ({f_min}-{f_max} THz)")
            except Exception as e:
                print(f"    [FAIL] Zoomed ({f_min}-{f_max} THz): {e}")
        
        # 6. Overlay with experiment
        if exp_data:
            try:
                self.plot_band_with_experiment(exp_data, save=True, output_dir=output_dir)
                print(f"    [OK] Theory vs experiment")
            except Exception as e:
                print(f"    [FAIL] Theory vs experiment: {e}")
        
        self.band_plotter.close_all()
        print(f"  [Bands] Generation complete")
    
    # ========================================================================
    # DOS PLOTS
    # ========================================================================
    
    def plot_dos(self, save: bool = False, output_dir: str = None,
                stacked: bool = False, unit: str = 'THz',
                elements: List[str] = None, **kwargs) -> Tuple:
        """Plot phonon density of states.
        
        Args:
            save: Save figure to file
            output_dir: Output directory
            stacked: Stack element contributions (True) or overlay (False)
            unit: Frequency unit ('THz', 'cm^-1', 'meV')
            elements: List of elements to show (None = all)
            **kwargs: Additional plot arguments
        
        Returns:
            (fig, axes) tuple
        """
        if not self.bands:
            raise ValueError("No DOS data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        freq, pdos_arr, sym_to_idx = pdos_data
        total_dos = pdos_arr.sum(axis=0)
        
        # Main DOS plot
        fig, ax = self.dos_plotter.plot_total_dos(
            freq, total_dos,
            formula=self.formula,
            unit=unit,
            title=kwargs.get('title', None),
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'phonon_dos_total.png')
            self.dos_plotter.save(fig, path)
        
        return fig, ax
    
    # ========================================================================
    # ADVANCED DOS PLOTS
    # ========================================================================
    
    def plot_partial_dos(self, save: bool = False, output_dir: str = None,
                        stacked: bool = False, unit: str = 'THz',
                        elements: List[str] = None, **kwargs) -> Tuple:
        """Plot element-projected partial DOS."""
        if not self.bands:
            raise ValueError("No DOS data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        freq, pdos_arr, sym_to_idx = pdos_data
        
        fig, ax = self.dos_plotter.plot_partial_dos(
            freq, pdos_arr, sym_to_idx,
            formula=self.formula,
            unit=unit,
            stacked=stacked,
            elements=elements,
            **kwargs
        )
        
        if save and output_dir:
            name = 'phonon_dos_stacked.png' if stacked else 'phonon_dos_partial.png'
            path = os.path.join(output_dir, name)
            self.dos_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_cumulative_dos(self, save: bool = False, output_dir: str = None,
                           unit: str = 'THz', **kwargs) -> Tuple:
        """Plot cumulative (integrated) DOS."""
        if not self.bands:
            raise ValueError("No DOS data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        freq, pdos_arr, sym_to_idx = pdos_data
        total_dos = pdos_arr.sum(axis=0)
        
        fig, ax = self.dos_plotter.plot_cumulative_dos(
            freq, total_dos,
            formula=self.formula,
            unit=unit,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'phonon_dos_cumulative.png')
            self.dos_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_mode_projected_dos(self, save: bool = False, output_dir: str = None,
                               acoustic_cutoff: float = 2.0, **kwargs) -> Tuple:
        """Plot mode-projected DOS (acoustic vs optical)."""
        if not self.bands:
            raise ValueError("No DOS data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        freq, pdos_arr, sym_to_idx = pdos_data
        
        fig, ax = self.dos_plotter.plot_mode_projected_dos(
            freq, pdos_arr, sym_to_idx,
            formula=self.formula,
            acoustic_cutoff=acoustic_cutoff,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'phonon_dos_mode_projected.png')
            self.dos_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_log_dos(self, save: bool = False, output_dir: str = None,
                    unit: str = 'THz', **kwargs) -> Tuple:
        """Plot DOS on logarithmic scale."""
        if not self.bands:
            raise ValueError("No DOS data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        freq, pdos_arr, sym_to_idx = pdos_data
        total_dos = pdos_arr.sum(axis=0)
        
        fig, ax = self.dos_plotter.plot_log_dos(
            freq, total_dos,
            formula=self.formula,
            unit=unit,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'phonon_dos_log.png')
            self.dos_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_hydrogen_dos(self, save: bool = False, output_dir: str = None,
                         unit: str = 'THz', **kwargs) -> Tuple:
        """Plot hydrogen-specific DOS with mode decomposition."""
        if not self.bands:
            raise ValueError("No DOS data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        freq, pdos_arr, sym_to_idx = pdos_data
        
        fig, ax = self.dos_plotter.plot_hydrogen_dos(
            freq, pdos_arr, sym_to_idx,
            formula=self.formula,
            unit=unit,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'H_dos.png')
            self.dos_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_h_mode_decomposition(self, save: bool = False, output_dir: str = None,
                                 unit: str = 'THz', **kwargs) -> Tuple:
        """Plot hydrogen mode decomposition."""
        if not self.bands:
            raise ValueError("No DOS data available")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        freq, pdos_arr, sym_to_idx = pdos_data
        
        if 'H' not in sym_to_idx:
            raise ValueError("No hydrogen atoms found in DOS data")
        
        h_indices = sym_to_idx['H']
        h_dos = np.sum([pdos_arr[i] for i in h_indices if i < pdos_arr.shape[0]], axis=0)
        
        fig, ax = self.dos_plotter.plot_h_mode_decomposition(
            freq, h_dos,
            formula=self.formula,
            unit=unit,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'H_mode_decomposition.png')
            self.dos_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_all_dos_types(self, output_dir: str):
        """Generate all DOS plot types."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"  [DOS] Generating all DOS plots in {output_dir}")
        
        if not self.bands or not self.bands.get('pdos_data'):
            print(f"    [SKIP] No pDOS data available")
            return
        
        # 1. Total DOS
        try:
            self.plot_total_dos(save=True, output_dir=output_dir)
            print(f"    [OK] Total DOS")
        except Exception as e:
            print(f"    [FAIL] Total DOS: {e}")
        
        # 2. Partial DOS (overlay)
        try:
            self.plot_partial_dos(save=True, output_dir=output_dir, stacked=False)
            print(f"    [OK] Partial DOS (overlay)")
        except Exception as e:
            print(f"    [FAIL] Partial DOS: {e}")
        
        # 3. Partial DOS (stacked)
        try:
            self.plot_partial_dos(save=True, output_dir=output_dir, stacked=True)
            print(f"    [OK] Partial DOS (stacked)")
        except Exception as e:
            print(f"    [FAIL] Partial DOS (stacked): {e}")
        
        # 4. Cumulative DOS
        try:
            self.plot_cumulative_dos(save=True, output_dir=output_dir)
            print(f"    [OK] Cumulative DOS")
        except Exception as e:
            print(f"    [FAIL] Cumulative DOS: {e}")
        
        # 5. Mode-projected DOS
        try:
            self.plot_mode_projected_dos(save=True, output_dir=output_dir)
            print(f"    [OK] Mode-projected DOS")
        except Exception as e:
            print(f"    [FAIL] Mode-projected DOS: {e}")
        
        # 6. Log-scale DOS
        try:
            self.plot_log_dos(save=True, output_dir=output_dir)
            print(f"    [OK] Log-scale DOS")
        except Exception as e:
            print(f"    [FAIL] Log-scale DOS: {e}")
        
        # 7. Hydrogen DOS (if H present)
        pdos_data = self.bands.get('pdos_data')
        if pdos_data:
            freq, pdos_arr, sym_to_idx = pdos_data
            if 'H' in sym_to_idx:
                try:
                    self.plot_hydrogen_dos(save=True, output_dir=output_dir)
                    print(f"    [OK] Hydrogen DOS")
                except Exception as e:
                    print(f"    [FAIL] Hydrogen DOS: {e}")
                
                try:
                    self.plot_h_mode_decomposition(save=True, output_dir=output_dir)
                    print(f"    [OK] H-mode decomposition")
                except Exception as e:
                    print(f"    [FAIL] H-mode decomposition: {e}")
        
        self.dos_plotter.close_all()
        print(f"  [DOS] Generation complete")
    
    # ========================================================================
    # THERMODYNAMIC PLOTS
    # ========================================================================
    
    def plot_thermodynamics(self, save: bool = False, output_dir: str = None,
                           show_dulong_petit: bool = True, **kwargs) -> Tuple:
        """Plot thermodynamic properties F(T), S(T), Cv(T) triple panel."""
        thermo = self._get_thermo_data()
        if len(thermo['temps']) == 0:
            raise ValueError("No thermodynamic data available")
        
        n_atoms = self._get_n_atoms()
        
        fig, axes = self.thermo_plotter.plot_triple_panel(
            thermo['temps'], thermo['free_energy'], thermo['entropy'], thermo['cv'],
            formula=self.formula,
            n_atoms=n_atoms,
            show_dulong_petit=show_dulong_petit,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'phonon_thermodynamics.png')
            self.thermo_plotter.save(fig, path)
        
        return fig, axes
    
    def plot_cv_dulong_petit(self, save: bool = False, output_dir: str = None,
                            **kwargs) -> Tuple:
        """Plot Cv(T) with Dulong-Petit validation."""
        thermo = self._get_thermo_data()
        if len(thermo['temps']) == 0:
            raise ValueError("No thermodynamic data available")
        
        n_atoms = self._get_n_atoms()
        
        fig, ax = self.thermo_plotter.plot_cv_dulong_petit(
            thermo['temps'], thermo['cv'],
            formula=self.formula,
            n_atoms=n_atoms,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'cv_dulong_petit.png')
            self.thermo_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_free_energy_components(self, save: bool = False, output_dir: str = None,
                                   **kwargs) -> Tuple:
        """Plot free energy decomposition into components."""
        thermo = self._get_thermo_data()
        if len(thermo['temps']) == 0:
            raise ValueError("No thermodynamic data available")
        
        fig, ax = self.thermo_plotter.plot_free_energy_components(
            thermo['temps'], thermo['free_energy'], thermo['entropy'],
            formula=self.formula,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'free_energy_components.png')
            self.thermo_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_cp_vs_cv(self, save: bool = False, output_dir: str = None,
                     **kwargs) -> Tuple:
        """Plot Cp vs Cv comparison."""
        thermo = self._get_thermo_data()
        if len(thermo['temps']) == 0:
            raise ValueError("No thermodynamic data available")
        
        # Estimate Cp from Cv (if not available, assume Cp ≈ Cv for solids)
        cv = thermo['cv']
        cp = kwargs.pop('cp', cv * 1.02)  # Default: Cp ≈ Cv + 2%
        
        fig, ax = self.thermo_plotter.plot_cp_vs_cv(
            thermo['temps'], cv, cp,
            formula=self.formula,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'cp_vs_cv.png')
            self.thermo_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_low_t_behavior(self, save: bool = False, output_dir: str = None,
                           debye_temp: float = None, **kwargs) -> Tuple:
        """Plot low-temperature Cv behavior validating T³ law."""
        thermo = self._get_thermo_data()
        if len(thermo['temps']) == 0:
            raise ValueError("No thermodynamic data available")
        
        fig, ax = self.thermo_plotter.plot_low_t_behavior(
            thermo['temps'], thermo['cv'],
            formula=self.formula,
            debye_temp=debye_temp,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'low_t_behavior.png')
            self.thermo_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_high_t_dulong_petit(self, save: bool = False, output_dir: str = None,
                                **kwargs) -> Tuple:
        """Plot high-temperature convergence to Dulong-Petit limit."""
        thermo = self._get_thermo_data()
        if len(thermo['temps']) == 0:
            raise ValueError("No thermodynamic data available")
        
        n_atoms = self._get_n_atoms()
        
        fig, ax = self.thermo_plotter.plot_high_t_dulong_petit(
            thermo['temps'], thermo['cv'],
            formula=self.formula,
            n_atoms=n_atoms,
            **kwargs
        )
        
        if save and output_dir:
            path = os.path.join(output_dir, 'high_t_dulong_petit.png')
            self.thermo_plotter.save(fig, path)
        
        return fig, ax
    
    def plot_all_thermo_types(self, output_dir: str, debye_temp: float = None):
        """Generate all thermodynamic plot types."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"  [Thermo] Generating all thermodynamic plots in {output_dir}")
        
        if not self.thermo or not self.thermo.get('thermo_data'):
            print(f"    [SKIP] No thermodynamic data available")
            return
        
        # 1. Triple panel
        try:
            self.plot_thermodynamics(save=True, output_dir=output_dir)
            print(f"    [OK] Triple panel (F/S/Cv)")
        except Exception as e:
            print(f"    [FAIL] Triple panel: {e}")
        
        # 2. Cv with Dulong-Petit
        try:
            self.plot_cv_dulong_petit(save=True, output_dir=output_dir)
            print(f"    [OK] Cv with Dulong-Petit")
        except Exception as e:
            print(f"    [FAIL] Cv with Dulong-Petit: {e}")
        
        # 3. Free energy components
        try:
            self.plot_free_energy_components(save=True, output_dir=output_dir)
            print(f"    [OK] Free energy components")
        except Exception as e:
            print(f"    [FAIL] Free energy components: {e}")
        
        # 4. Cp vs Cv
        try:
            self.plot_cp_vs_cv(save=True, output_dir=output_dir)
            print(f"    [OK] Cp vs Cv")
        except Exception as e:
            print(f"    [FAIL] Cp vs Cv: {e}")
        
        # 5. Low-T behavior
        try:
            self.plot_low_t_behavior(save=True, output_dir=output_dir, debye_temp=debye_temp)
            print(f"    [OK] Low-T behavior")
        except Exception as e:
            print(f"    [FAIL] Low-T behavior: {e}")
        
        # 6. High-T Dulong-Petit
        try:
            self.plot_high_t_dulong_petit(save=True, output_dir=output_dir)
            print(f"    [OK] High-T Dulong-Petit")
        except Exception as e:
            print(f"    [FAIL] High-T Dulong-Petit: {e}")
        
        self.thermo_plotter.close_all()
        print(f"  [Thermo] Generation complete")
    
    # ========================================================================
    # HYDROGEN PLOTS
    # ========================================================================
    
    def plot_hydrogen_analysis(self, save: bool = False, output_dir: str = None,
                              **kwargs) -> Tuple:
        """Plot hydrogen vibrational mode decomposition.
        
        Args:
            save: Save figure to file
            output_dir: Output directory
            **kwargs: Additional plot arguments
        
        Returns:
            (fig, ax) tuple
        """
        if not self.hydrogen:
            raise ValueError("No hydrogen analysis data available")
        
        if not self.bands:
            raise ValueError("No pDOS data available for H analysis")
        
        pdos_data = self.bands.get('pdos_data')
        if not pdos_data:
            raise ValueError("No pDOS data available")
        
        import matplotlib.pyplot as plt
        fig, ax = self.plotter.create_figure(figsize=(10, 5))
        
        freq, pdos_arr, sym_to_idx = pdos_data
        
        h_dos = np.zeros_like(freq)
        for idx in sym_to_idx.get('H', []):
            if idx < pdos_arr.shape[0]:
                h_dos += pdos_arr[idx]
        
        lib = (freq > 5) & (freq < 20.9)
        bend = (freq > 20.9) & (freq < 50)
        stretch = (freq > 50) & (freq < 100)
        
        ax.fill_between(freq[lib], h_dos[lib], alpha=0.3, color='#9B59B6', label='Librational')
        ax.fill_between(freq[bend], h_dos[bend], alpha=0.3, color='#3498DB', label='Bending')
        ax.fill_between(freq[stretch], h_dos[stretch], alpha=0.3, color='#E74C3C', label='H Stretching')
        ax.plot(freq, h_dos, color='#F39C12', lw=1.5, label='H total')
        
        # Peak marker
        peak = self.hydrogen.get('peak_stretching')
        if peak:
            ax.axvline(x=peak['freq_thz'], color='#E74C3C', ls='--', alpha=0.7)
            ax.text(peak['freq_thz'] + 1, max(h_dos) * 0.85,
                   f"Peak\n{peak['freq_thz']:.1f} THz\n({peak['freq_cm']:.0f} cm^-1)",
                   fontsize=9, color='#E74C3C', fontweight='bold')
        
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('H Partial DOS')
        ax.set_title(f'{self.formula} - Hydrogen Mode Decomposition', fontweight='bold')
        ax.legend()
        ax.grid(alpha=self.plotter.theme['axes']['grid_alpha'])
        
        pos_mask = h_dos > 0.001
        if np.any(pos_mask):
            ax.set_xlim(0, max(freq[pos_mask]) * 1.05)
        
        fig.tight_layout()
        
        if save and output_dir:
            path = os.path.join(output_dir, 'H_mode_analysis.png')
            self.plotter.save(fig, path)
        
        return fig, ax

    # ========================================================================
    # IFC & BORN PLOTS
    # ========================================================================
    
    def plot_ifc_decay(self, distances: np.ndarray, ifc_magnitudes: np.ndarray,
                      save: bool = False, output_dir: str = None,
                      **kwargs) -> Tuple:
        """Plot IFC magnitude vs interatomic distance."""
        fig, ax = self.ifc_born_plotter.plot_ifc_decay(
            distances, ifc_magnitudes, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'ifc_decay.png')
            self.ifc_born_plotter.save(fig, path)
        return fig, ax
    
    def plot_ifc_heatmap(self, ifc_matrix: np.ndarray,
                        save: bool = False, output_dir: str = None,
                        **kwargs) -> Tuple:
        """Plot force constant matrix heatmap."""
        fig, ax = self.ifc_born_plotter.plot_ifc_heatmap(
            ifc_matrix, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'ifc_heatmap.png')
            self.ifc_born_plotter.save(fig, path)
        return fig, ax
    
    def plot_born_charges(self, born_charges: np.ndarray,
                         save: bool = False, output_dir: str = None,
                         **kwargs) -> Tuple:
        """Plot Born effective charge tensors."""
        fig, ax = self.ifc_born_plotter.plot_born_charges(
            born_charges, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'born_charges.png')
            self.ifc_born_plotter.save(fig, path)
        return fig, ax
    
    def plot_dielectric_tensor(self, dielectric: np.ndarray,
                              save: bool = False, output_dir: str = None,
                              **kwargs) -> Tuple:
        """Plot dielectric tensor heatmap."""
        fig, ax = self.ifc_born_plotter.plot_dielectric_tensor(
            dielectric, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'dielectric_tensor.png')
            self.ifc_born_plotter.save(fig, path)
        return fig, ax
    
    def plot_charge_neutrality(self, born_charges: np.ndarray,
                              save: bool = False, output_dir: str = None,
                              **kwargs) -> Tuple:
        """Plot charge neutrality check."""
        fig, ax = self.ifc_born_plotter.plot_charge_neutrality(
            born_charges, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'charge_neutrality.png')
            self.ifc_born_plotter.save(fig, path)
        return fig, ax
    
    # ========================================================================
    # GRUNEISEN & TRANSPORT PLOTS
    # ========================================================================
    
    def plot_mode_gruneisen(self, freq: np.ndarray, gruneisen: np.ndarray,
                           save: bool = False, output_dir: str = None,
                           **kwargs) -> Tuple:
        """Plot mode Grüneisen parameters."""
        fig, ax = self.transport_plotter.plot_mode_gruneisen(
            freq, gruneisen, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'mode_gruneisen.png')
            self.transport_plotter.save(fig, path)
        return fig, ax
    
    def plot_average_gruneisen(self, temps: np.ndarray, avg_gruneisen: np.ndarray,
                              save: bool = False, output_dir: str = None,
                              **kwargs) -> Tuple:
        """Plot average Grüneisen parameter vs temperature."""
        fig, ax = self.transport_plotter.plot_average_gruneisen(
            temps, avg_gruneisen, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'average_gruneisen.png')
            self.transport_plotter.save(fig, path)
        return fig, ax
    
    def plot_gruneisen_distribution(self, gruneisen: np.ndarray,
                                   save: bool = False, output_dir: str = None,
                                   **kwargs) -> Tuple:
        """Plot Grüneisen parameter distribution."""
        fig, ax = self.transport_plotter.plot_gruneisen_distribution(
            gruneisen, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'gruneisen_distribution.png')
            self.transport_plotter.save(fig, path)
        return fig, ax
    
    def plot_thermal_conductivity(self, temps: np.ndarray, kappa: np.ndarray,
                                 save: bool = False, output_dir: str = None,
                                 **kwargs) -> Tuple:
        """Plot lattice thermal conductivity κ(T)."""
        fig, ax = self.transport_plotter.plot_thermal_conductivity(
            temps, kappa, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'thermal_conductivity.png')
            self.transport_plotter.save(fig, path)
        return fig, ax
    
    def plot_cumulative_kappa(self, mfp: np.ndarray, kappa_cumulative: np.ndarray,
                             save: bool = False, output_dir: str = None,
                             **kwargs) -> Tuple:
        """Plot cumulative thermal conductivity vs MFP."""
        fig, ax = self.transport_plotter.plot_cumulative_kappa(
            mfp, kappa_cumulative, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'cumulative_kappa.png')
            self.transport_plotter.save(fig, path)
        return fig, ax
    
    def plot_kappa_decomposition(self, freq: np.ndarray, kappa_by_freq: np.ndarray,
                                save: bool = False, output_dir: str = None,
                                **kwargs) -> Tuple:
        """Plot thermal conductivity decomposition by frequency."""
        fig, ax = self.transport_plotter.plot_kappa_decomposition(
            freq, kappa_by_freq, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'kappa_decomposition.png')
            self.transport_plotter.save(fig, path)
        return fig, ax
    
    def plot_slack_comparison(self, temps: np.ndarray, kappa_calc: np.ndarray,
                             theta_D: float, avg_gamma: float,
                             save: bool = False, output_dir: str = None,
                             **kwargs) -> Tuple:
        """Plot calculated κ vs Slack model prediction."""
        fig, ax = self.transport_plotter.plot_slack_comparison(
            temps, kappa_calc, theta_D, avg_gamma, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'slack_comparison.png')
            self.transport_plotter.save(fig, path)
        return fig, ax

    # ========================================================================
    # HYDROGEN PLOTS
    # ========================================================================
    
    def plot_h_dos_with_modes(self, freq: np.ndarray, h_dos: np.ndarray,
                             save: bool = False, output_dir: str = None,
                             **kwargs) -> Tuple:
        """Plot hydrogen DOS with mode region labels."""
        fig, ax = self.hydrogen_plotter.plot_h_dos_with_modes(
            freq, h_dos, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'H_dos_with_modes.png')
            self.hydrogen_plotter.save(fig, path)
        return fig, ax
    
    def plot_h_stretch_zoom(self, freq: np.ndarray, h_dos: np.ndarray,
                           save: bool = False, output_dir: str = None,
                           **kwargs) -> Tuple:
        """Plot zoomed view of H stretching region."""
        fig, ax = self.hydrogen_plotter.plot_stretch_zoom(
            freq, h_dos, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'H_stretch_zoom.png')
            self.hydrogen_plotter.save(fig, path)
        return fig, ax
    
    def plot_h_mode_pie(self, freq: np.ndarray, h_dos: np.ndarray,
                       save: bool = False, output_dir: str = None,
                       **kwargs) -> Tuple:
        """Plot hydrogen mode fractions as pie chart."""
        fig, ax = self.hydrogen_plotter.plot_h_mode_pie_chart(
            freq, h_dos, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'H_mode_pie.png')
            self.hydrogen_plotter.save(fig, path)
        return fig, ax
    
    def plot_hydride_type(self, peak_freq_cm: float,
                         save: bool = False, output_dir: str = None,
                         **kwargs) -> Tuple:
        """Plot hydride type identification."""
        fig, ax = self.hydrogen_plotter.plot_hydride_type_comparison(
            peak_freq_cm, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'hydride_type.png')
            self.hydrogen_plotter.save(fig, path)
        return fig, ax
    
    # ========================================================================
    # COMPARISON PLOTS
    # ========================================================================
    
    def plot_multi_material_bands(self, band_data: Dict[str, Dict],
                                 save: bool = False, output_dir: str = None,
                                 **kwargs) -> Tuple:
        """Plot multiple materials' band structures overlaid."""
        fig, ax = self.comparison_plotter.plot_multi_material_bands(
            band_data, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'multi_material_bands.png')
            self.comparison_plotter.save(fig, path)
        return fig, ax
    
    def plot_multi_material_dos(self, dos_data: Dict[str, Tuple],
                               save: bool = False, output_dir: str = None,
                               **kwargs) -> Tuple:
        """Plot multiple materials' DOS overlaid."""
        fig, ax = self.comparison_plotter.plot_multi_material_dos(
            dos_data, formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'multi_material_dos.png')
            self.comparison_plotter.save(fig, path)
        return fig, ax
    
    def plot_theory_vs_experiment(self, calc_freq: np.ndarray,
                                 calc_intensity: np.ndarray,
                                 exp_freq: np.ndarray,
                                 exp_intensity: np.ndarray,
                                 exp_errors: np.ndarray = None,
                                 save: bool = False, output_dir: str = None,
                                 **kwargs) -> Tuple:
        """Plot calculated vs experimental spectra."""
        fig, ax = self.comparison_plotter.plot_theory_vs_experiment(
            calc_freq, calc_intensity, exp_freq, exp_intensity, exp_errors,
            formula=self.formula, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'theory_vs_experiment.png')
            self.comparison_plotter.save(fig, path)
        return fig, ax
    
    def plot_convergence_supercell(self, supercell_sizes: List[str],
                                  frequencies: List[np.ndarray],
                                  save: bool = False, output_dir: str = None,
                                  **kwargs) -> Tuple:
        """Plot convergence vs supercell size."""
        fig, ax = self.comparison_plotter.plot_convergence_supercell(
            supercell_sizes, frequencies, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'convergence_supercell.png')
            self.comparison_plotter.save(fig, path)
        return fig, ax
    
    def plot_convergence_qmesh(self, mesh_sizes: List[str],
                              property_values: List[float],
                              save: bool = False, output_dir: str = None,
                              **kwargs) -> Tuple:
        """Plot convergence vs q-mesh density."""
        fig, ax = self.comparison_plotter.plot_convergence_qmesh(
            mesh_sizes, property_values, **kwargs)
        
        if save and output_dir:
            path = os.path.join(output_dir, 'convergence_qmesh.png')
            self.comparison_plotter.save(fig, path)
        return fig, ax

    # ========================================================================
    # BULK GENERATION
    # ========================================================================
    
    def generate_all(self, output_dir: str, formats: List[str] = None,
                    skip: List[str] = None):
        """Generate all available figures.
        
        Args:
            output_dir: Output directory
            formats: List of formats to save (defaults to theme format)
            skip: List of plot types to skip
        """
        os.makedirs(output_dir, exist_ok=True)
        
        skip = skip or []
        formats = formats or [self.plotter.theme['savefig']['format']]
        
        print(f"  [Figures] Generating all plots in {output_dir}")
        
        # Band Structure
        if 'bands' not in skip and self.bands:
            try:
                self.plot_band_structure(save=True, output_dir=output_dir)
                print(f"    [OK] Band structure")
            except Exception as e:
                print(f"    [FAIL] Band structure: {e}")
        
        # Fat Bands
        if 'fat_bands' not in skip and self.bands and self.bands.get('pdos_data'):
            try:
                self.plot_fat_bands(save=True, output_dir=output_dir)
                print(f"    [OK] Fat bands")
            except Exception as e:
                print(f"    [FAIL] Fat bands: {e}")
        
        # DOS
        if 'dos' not in skip and self.bands and self.bands.get('pdos_data'):
            try:
                self.plot_dos(save=True, output_dir=output_dir, stacked=True)
                print(f"    [OK] DOS")
            except Exception as e:
                print(f"    [FAIL] DOS: {e}")
        
        # Thermodynamics
        if 'thermo' not in skip and self.thermo and self.thermo.get('thermo_data'):
            try:
                self.plot_thermodynamics(save=True, output_dir=output_dir)
                print(f"    [OK] Thermodynamics")
            except Exception as e:
                print(f"    [FAIL] Thermodynamics: {e}")
        
        # Hydrogen
        if 'hydrogen' not in skip and self.hydrogen and self.bands and self.bands.get('pdos_data'):
            try:
                self.plot_hydrogen_analysis(save=True, output_dir=output_dir)
                print(f"    [OK] Hydrogen analysis")
            except Exception as e:
                print(f"    [FAIL] Hydrogen analysis: {e}")
        
        # Save all tracked figures
        self.plotter.save_all(output_dir)
        self.plotter.close_all()
        
        print(f"  [Figures] Generation complete")
