"""
HydroPhonoKit Visualization — Publication-Ready Figure Generation

Provides comprehensive visualization capabilities for phonon analysis:
  - 6 publication-ready themes (Nature, Science, PRL, ACS, Presentation, Minimal)
  - Band structure plots (standard, fat bands)
  - Density of states plots (total, partial, stacked)
  - Thermodynamic property plots (F, S, Cv)
  - Hydrogen mode decomposition
  - Multi-format export (PNG, PDF, EPS, SVG)
  - Figure factory for easy batch generation

Usage:
    from hydrophonokit.visualization import PhononFigureFactory
    
    # Create factory with results and theme
    factory = PhononFigureFactory(results, theme='nature')
    
    # Generate specific plots
    factory.plot_band_structure(save=True, output_dir='figures/')
    factory.plot_dos(save=True, output_dir='figures/')
    
    # Or generate all at once
    factory.generate_all(output_dir='figures/')

References:
    [1] Nature Publishing Guide -- Figure preparation standards
    [2] APS Style Guide -- PRL figure requirements
    [3] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
"""

__version__ = "1.0.0"

# Public API
from .themes import (
    THEMES,
    DEFAULT_THEME,
    ELEMENT_COLORS,
    get_theme,
    list_themes,
    get_element_color,
)
from .base_plotter import (
    BasePlotter,
    THZ_TO_CM,
    THZ_TO_MEV,
    CM_TO_THZ,
    MEV_TO_THZ,
    freq_to_cm,
    freq_to_meV,
)
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
from .figure_factory import PhononFigureFactory

__all__ = [
    # Core
    'PhononFigureFactory',
    'BasePlotter',
    'BandStructurePlotter',
    'DOSPlotter',
    'ThermoPlotter',
    'IFCBornPlotter',
    'TransportPlotter',
    'HydrogenPlotter',
    'ComparisonPlotter',
    'InteractivePlotter',
    'PublicationFormatter',
    'MultiPanelComposer',
    # Themes
    'THEMES',
    'DEFAULT_THEME',
    'ELEMENT_COLORS',
    'get_theme',
    'list_themes',
    'get_element_color',
    # Unit conversions
    'THZ_TO_CM',
    'THZ_TO_MEV',
    'CM_TO_THZ',
    'MEV_TO_THZ',
    'freq_to_cm',
    'freq_to_meV',
]
