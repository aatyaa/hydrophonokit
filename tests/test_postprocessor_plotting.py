"""
Tests for HydroPhonoKit Postprocessor visualization.

Run with: pytest tests/test_postprocessor_plotting.py -v
"""
import pytest
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test Themes
# ============================================================================

class TestThemes:
    """Test theme definitions and application."""

    def test_nature_theme_exists(self):
        """Nature theme should be defined."""
        from hydrophonokit.postprocessor.advanced_plotting import THEMES
        assert 'nature' in THEMES

    def test_science_theme_exists(self):
        """Science theme should be defined."""
        from hydrophonokit.postprocessor.advanced_plotting import THEMES
        assert 'science' in THEMES

    def test_dark_theme_exists(self):
        """Dark theme should be defined."""
        from hydrophonokit.postprocessor.advanced_plotting import THEMES
        assert 'dark' in THEMES

    def test_minimal_theme_exists(self):
        """Minimal theme should be defined."""
        from hydrophonokit.postprocessor.advanced_plotting import THEMES
        assert 'minimal' in THEMES

    def test_theme_has_required_keys(self):
        """Each theme should have required configuration keys."""
        from hydrophonokit.postprocessor.advanced_plotting import THEMES
        required = {'font', 'figure', 'axes', 'colors', 'legend', 'savefig'}

        for name, theme in THEMES.items():
            for key in required:
                assert key in theme, f"Theme '{name}' missing key '{key}'"

    def test_element_colors_defined(self):
        """ELEMENT_COLORS should be defined with common elements."""
        from hydrophonokit.postprocessor.advanced_plotting import ELEMENT_COLORS
        assert 'H' in ELEMENT_COLORS
        assert 'Mg' in ELEMENT_COLORS
        assert 'Si' in ELEMENT_COLORS
        assert 'default' in ELEMENT_COLORS


# ============================================================================
# Test Advanced Plotter
# ============================================================================

class TestAdvancedPlotter:
    """Test AdvancedPhononPlotter functionality."""

    def _make_mock_band_dict(self):
        """Create a mock band structure dictionary."""
        dist = np.linspace(0, 1, 50)
        freq = np.ones((50, 6)) * np.linspace(2, 20, 50)[:, np.newaxis]
        return {
            'distances': [dist, dist],
            'frequencies': [freq, freq],
        }

    def _make_mock_pdos(self):
        """Create mock pDOS data."""
        freq = np.linspace(0, 30, 300)
        pdos = np.random.randn(3, 300) ** 2
        sym_to_idx = {'Mg': [0], 'H': [1, 2]}
        return freq, pdos, sym_to_idx

    def test_init_default_theme(self):
        """Should initialize with 'nature' theme by default."""
        from hydrophonokit.postprocessor.advanced_plotting import PhononPlotter
        plotter = PhononPlotter()
        assert plotter.theme_name == 'nature'

    def test_init_custom_theme(self):
        """Should initialize with specified theme."""
        from hydrophonokit.postprocessor.advanced_plotting import PhononPlotter
        plotter = PhononPlotter(theme='dark')
        assert plotter.theme_name == 'dark'

    def test_plot_band_structure_returns_fig(self):
        """plot_band_structure should return (fig, ax) tuple."""
        from hydrophonokit.postprocessor.advanced_plotting import PhononPlotter
        plotter = PhononPlotter(theme='nature')
        band_dict = self._make_mock_band_dict()
        fig, ax = plotter.plot_band_structure(band_dict, 'TestMaterial')

        assert fig is not None
        assert ax is not None

    def test_plot_partial_dos_returns_fig(self):
        """plot_partial_dos should return (fig, axes) tuple."""
        from hydrophonokit.postprocessor.advanced_plotting import PhononPlotter
        plotter = PhononPlotter()
        pdos_data = self._make_mock_pdos()
        fig, axes = plotter.plot_partial_dos(pdos_data, 'MgH2')

        assert fig is not None
        assert axes is not None

    def test_plot_thermodynamics_returns_fig(self):
        """plot_thermodynamics should return (fig, axes) tuple."""
        from hydrophonokit.postprocessor.advanced_plotting import PhononPlotter
        plotter = PhononPlotter()

        thermo_data = {
            'temperatures': np.arange(0, 1001, 10),
            'free_energy': np.linspace(-100, -150, 101),
            'entropy': np.linspace(0, 100, 101),
            'heat_capacity': np.linspace(0, 70, 101),
        }
        fig, axes = plotter.plot_thermodynamics(thermo_data, 'Test', 6)

        assert fig is not None
        assert axes is not None


# ============================================================================
# Test Specialized Plots
# ============================================================================

class TestSpecializedPlots:
    """Test specialized plot types."""

    def test_cumulative_kappa_returns_fig(self):
        """Cumulative kappa plot should return (fig, ax)."""
        from hydrophonokit.postprocessor.specialized_plots import SpecializedPlots

        mfp = np.logspace(-1, 3, 100)  # 0.1 to 1000 nm
        kappa = np.ones(100) * 0.1
        fig, ax = SpecializedPlots.plot_cumulative_kappa(mfp, kappa, 'Si')

        assert fig is not None
        assert ax is not None

    def test_convergence_plot_returns_fig(self):
        """Convergence plot should return (fig, ax)."""
        from hydrophonokit.postprocessor.specialized_plots import SpecializedPlots

        mesh_sizes = [5, 8, 10, 12, 15]
        freqs = [15.2, 15.5, 15.6, 15.65, 15.66]
        fig, ax = SpecializedPlots.plot_convergence_mesh(mesh_sizes, freqs)

        assert fig is not None
        assert ax is not None

    def test_multi_overlay_returns_fig(self):
        """Multi-material overlay should return (fig, ax)."""
        from hydrophonokit.postprocessor.specialized_plots import SpecializedPlots

        band1 = {
            'distances': [np.linspace(0, 1, 30)],
            'frequencies': [np.ones((30, 3)) * np.linspace(2, 15, 30)[:, np.newaxis]],
        }
        band2 = {
            'distances': [np.linspace(0, 1, 30)],
            'frequencies': [np.ones((30, 3)) * np.linspace(3, 18, 30)[:, np.newaxis]],
        }
        fig, ax = SpecializedPlots.plot_multi_material_overlay(
            {'Si': band1, 'Ge': band2}, ['Si', 'Ge']
        )

        assert fig is not None
        assert ax is not None
