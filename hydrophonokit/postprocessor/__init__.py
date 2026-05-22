"""
HydroPhonoKit Postprocessor — Scientific Phonon Analysis Engine

This package provides rigorous post-processing capabilities for phonon
calculations, including:
  - Force constant computation (symfc/phonopy)
  - Band structure and density of states
  - Thermodynamic properties (F, S, Cv)
  - Hydrogen storage analytics
  - Publication-quality visualization
  - Standardized data export

Scientific Foundation:
  The postprocessor implements the standard workflow for phonon analysis
  from first-principles calculations, respecting:
    - Acoustic Sum Rules (ASR) for force constants
    - Non-Analytical Corrections (NAC) for LO-TO splitting
    - Born-Huang dynamical stability criteria
    - Harmonic approximation thermodynamics
    - Hydrogen vibrational mode decomposition

References:
  [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
  [2] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT
  [3] Born & Huang, Dynamical Theory of Crystal Lattices (1954)
  [4] Wang et al., Phys. Rev. B 95, 014303 (2017) -- symfc
  [5] Gonze & Lee, Phys. Rev. B 55, 10355 (1997) -- LO-TO splitting
"""

__version__ = "2.7.0"

# Public API
from .core import PhononPostProcessor, PostprocessorConfig, VALID_PHASES
from .plotting import PhononPlotter
from .advanced_plotting import PhononPlotter as AdvancedPhononPlotter, THEMES, ELEMENT_COLORS
from .specialized_plots import SpecializedPlots
from .enhanced_export import (
    PhononResultsExporter, 
    EXPORT_TEMPLATES, 
    ExportValidator,
    ExportTemplate,
)
from .caching import CacheManager, CacheConfig, MemoryMonitor, get_n_jobs
from .data_loader import DataLoader
from .ifc_computer import IFCComputer
from .bands_dos import BandsDOSComputer
from .thermodynamics import ThermodynamicsComputer
from .hydrogen import HydrogenAnalyzer
from .group_velocities import GroupVelocityComputer
from .debye_waller import DebyeWallerComputer
from .mode_resolved_thermo import ModeResolvedThermo
from .reporting import ReportGenerator

__all__ = [
    'PhononPostProcessor',
    'PostprocessorConfig',
    'VALID_PHASES',
    'PhononPlotter',
    'AdvancedPhononPlotter',
    'THEMES',
    'ELEMENT_COLORS',
    'SpecializedPlots',
    'PhononResultsExporter',
    'EXPORT_TEMPLATES',
    'ExportValidator',
    'ExportTemplate',
    'CacheManager',
    'CacheConfig',
    'MemoryMonitor',
    'get_n_jobs',
    'DataLoader',
    'IFCComputer',
    'BandsDOSComputer',
    'ThermodynamicsComputer',
    'HydrogenAnalyzer',
    'GroupVelocityComputer',
    'DebyeWallerComputer',
    'ModeResolvedThermo',
    'ReportGenerator',
]
