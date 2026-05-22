"""
Tests for HydroPhonoKit Visualization Core Infrastructure.

Run with: pytest tests/test_visualization_core.py -v
"""
import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch


# ============================================================================
# Test Themes
# ============================================================================

class TestThemes:
    """Test theme definitions and management."""

    def test_all_themes_defined(self):
        """All expected themes should be defined."""
        from hydrophonokit.visualization.themes import THEMES
        expected = ['nature', 'science', 'prl', 'acs', 'presentation', 'minimal']
        for name in expected:
            assert name in THEMES, f"Theme '{name}' not found"

    def test_theme_has_required_keys(self):
        """Each theme should have all required configuration keys."""
        from hydrophonokit.visualization.themes import THEMES
        required_keys = ['description', 'font', 'figure', 'axes', 'colors', 
                        'legend', 'savefig', 'lines']
        
        for name, theme in THEMES.items():
            for key in required_keys:
                assert key in theme, f"Theme '{name}' missing key '{key}'"

    def test_get_theme_returns_valid_theme(self):
        """get_theme should return a valid theme dict."""
        from hydrophonokit.visualization.themes import get_theme
        theme = get_theme('nature')
        assert isinstance(theme, dict)
        assert 'colors' in theme
        assert 'font' in theme

    def test_get_theme_raises_for_unknown(self):
        """get_theme should raise ValueError for unknown theme."""
        from hydrophonokit.visualization.themes import get_theme
        with pytest.raises(ValueError, match="Unknown theme"):
            get_theme('nonexistent')

    def test_list_themes(self):
        """list_themes should return list of (name, description) tuples."""
        from hydrophonokit.visualization.themes import list_themes
        themes = list_themes()
        assert len(themes) >= 6
        for name, desc in themes:
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_element_colors_defined(self):
        """ELEMENT_COLORS should include common elements."""
        from hydrophonokit.visualization.themes import ELEMENT_COLORS
        common = ['H', 'Li', 'B', 'C', 'N', 'O', 'Mg', 'Si', 'Ca', 'Fe']
        for elem in common:
            assert elem in ELEMENT_COLORS, f"Element '{elem}' not in COLORS"

    def test_get_element_color(self):
        """get_element_color should return color for known and unknown elements."""
        from hydrophonokit.visualization.themes import get_element_color
        assert get_element_color('H') == '#F39C12'
        assert get_element_color('Mg') == '#2ECC71'
        # Unknown element should return default color
        assert get_element_color('Xx') == '#888888'


# ============================================================================
# Test Unit Conversions
# ============================================================================

class TestUnitConversions:
    """Test frequency unit conversions."""

    def test_thz_to_cm(self):
        """THz to cm^-1 conversion."""
        from hydrophonokit.visualization.base_plotter import freq_to_cm, THZ_TO_CM
        assert abs(freq_to_cm(1.0) - THZ_TO_CM) < 1e-10
        assert abs(freq_to_cm(10.0) - 333.564) < 0.01

    def test_thz_to_meV(self):
        """THz to meV conversion."""
        from hydrophonokit.visualization.base_plotter import freq_to_meV, THZ_TO_MEV
        assert abs(freq_to_meV(1.0) - THZ_TO_MEV) < 1e-10
        assert abs(freq_to_meV(10.0) - 41.357) < 0.01

    def test_cm_to_thz(self):
        """cm^-1 to THz conversion."""
        from hydrophonokit.visualization.base_plotter import CM_TO_THZ
        assert abs(33.356 * CM_TO_THZ - 1.0) < 0.001

    def test_meV_to_thz(self):
        """meV to THz conversion."""
        from hydrophonokit.visualization.base_plotter import MEV_TO_THZ
        assert abs(4.136 * MEV_TO_THZ - 1.0) < 0.001


# ============================================================================
# Test BasePlotter
# ============================================================================

class TestBasePlotter:
    """Test BasePlotter class."""

    def test_init_default_theme(self):
        """Should initialize with default theme."""
        from hydrophonokit.visualization.base_plotter import BasePlotter
        plotter = BasePlotter()
        assert plotter.theme_name == 'nature'

    def test_init_custom_theme(self):
        """Should initialize with specified theme."""
        from hydrophonokit.visualization.base_plotter import BasePlotter
        plotter = BasePlotter(theme='prl')
        assert plotter.theme_name == 'prl'

    def test_create_figure(self):
        """Should create figure and axes."""
        from hydrophonokit.visualization.base_plotter import BasePlotter
        plotter = BasePlotter()
        fig, ax = plotter.create_figure(figsize=(8, 6))
        assert fig is not None
        assert ax is not None
        plotter.close_all()

    def test_get_color(self):
        """Should return color from theme."""
        from hydrophonokit.visualization.base_plotter import BasePlotter
        plotter = BasePlotter(theme='nature')
        assert plotter.get_color('primary') == '#1E3A8A'
        assert plotter.get_color('unknown') == '#888888'

    def test_get_element_color(self):
        """Should return element color."""
        from hydrophonokit.visualization.base_plotter import BasePlotter
        plotter = BasePlotter()
        assert plotter.get_element_color('H') == '#F39C12'
        assert plotter.get_element_color('Mg') == '#2ECC71'

    def test_save_figure(self):
        """Should save figure to file."""
        from hydrophonokit.visualization.base_plotter import BasePlotter
        plotter = BasePlotter()
        fig, ax = plotter.create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.png')
            plotter.save(fig, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        
        plotter.close_all()

    def test_track_figures(self):
        """Should track created figures."""
        from hydrophonokit.visualization.base_plotter import BasePlotter
        plotter = BasePlotter()
        fig1, _ = plotter.create_figure()
        fig2, _ = plotter.create_figure()
        
        assert len(plotter.get_figures()) == 2
        assert fig1 in plotter.get_figures()
        assert fig2 in plotter.get_figures()
        
        plotter.close_all()
        assert len(plotter.get_figures()) == 0


# ============================================================================
# Test PhononFigureFactory
# ============================================================================

class TestPhononFigureFactory:
    """Test PhononFigureFactory class."""

    def _make_mock_results(self):
        """Create mock results dictionary."""
        profile = Mock()
        profile.formula = "TestMaterial"
        profile.space_group = "Pm-3m (221)"
        profile.n_atoms = 2

        # Mock band structure data
        dist = np.linspace(0, 1, 50)
        freq = np.ones((50, 6)) * np.linspace(2, 20, 50)[:, np.newaxis]
        freq[:, 0] = np.linspace(0, 5, 50)  # Acoustic mode
        
        band_dict = {
            'distances': [dist, dist],
            'frequencies': [freq, freq],
        }
        
        pdos_freq = np.linspace(0, 25, 250)
        pdos = np.random.randn(2, 250) ** 2  # 2 atoms
        
        return {
            'profile': profile,
            'data_loader': {'phonon': None},
            'bands_dos': {
                'band_dict': band_dict,
                'pdos_data': (pdos_freq, pdos, {'A': [0], 'B': [1]}),
                'min_freq': 0.0,
                'stability': {'stable': True, 'n_imaginary': 0},
            },
            'thermodynamics': {
                'zpe': 10.5,
                'at_300K': {'F_kJ_mol': -5.0, 'S_J_molK': 50.0, 'Cv_J_molK': 25.0},
                'thermo_data': {
                    'temperatures': np.arange(0, 1001, 10),
                    'free_energy': np.linspace(-10, -15, 101),
                    'entropy': np.linspace(0, 100, 101),
                    'heat_capacity': np.linspace(0, 70, 101),
                },
                'validations': {'third_law_ok': True},
            },
            'hydrogen': {},
        }

    def test_init(self):
        """Should initialize with results."""
        from hydrophonokit.visualization.figure_factory import PhononFigureFactory
        results = self._make_mock_results()
        factory = PhononFigureFactory(results, theme='nature')
        assert factory.formula == "TestMaterial"

    def test_plot_band_structure(self):
        """Should create band structure plot."""
        from hydrophonokit.visualization.figure_factory import PhononFigureFactory
        results = self._make_mock_results()
        factory = PhononFigureFactory(results)
        fig, ax = factory.plot_band_structure()
        assert fig is not None
        assert ax is not None
        factory.plotter.close_all()

    def test_plot_fat_bands(self):
        """Should create fat band structure plot."""
        from hydrophonokit.visualization.figure_factory import PhononFigureFactory
        results = self._make_mock_results()
        factory = PhononFigureFactory(results)
        fig, ax = factory.plot_fat_bands()
        assert fig is not None
        assert ax is not None
        factory.plotter.close_all()

    def test_plot_dos(self):
        """Should create DOS plot."""
        from hydrophonokit.visualization.figure_factory import PhononFigureFactory
        results = self._make_mock_results()
        factory = PhononFigureFactory(results)
        fig, axes = factory.plot_dos()
        assert fig is not None
        assert axes is not None
        factory.plotter.close_all()

    def test_plot_thermodynamics(self):
        """Should create thermodynamics plot."""
        from hydrophonokit.visualization.figure_factory import PhononFigureFactory
        results = self._make_mock_results()
        factory = PhononFigureFactory(results)
        fig, axes = factory.plot_thermodynamics()
        assert fig is not None
        assert axes is not None
        factory.plotter.close_all()

    def test_save_plots(self):
        """Should save plots to files."""
        from hydrophonokit.visualization.figure_factory import PhononFigureFactory
        results = self._make_mock_results()
        factory = PhononFigureFactory(results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            factory.plot_band_structure(save=True, output_dir=tmpdir)
            path = os.path.join(tmpdir, 'phonon_band_structure.png')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        
        factory.plotter.close_all()
