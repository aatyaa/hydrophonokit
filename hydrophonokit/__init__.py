__version__ = "2.7.0"

# Expose key physical constants for external use
from .physics import (
    # Fundamental constants
    H_PLANCK, K_BOLTZMANN, N_AVOGADRO, R_GAS, HBAR,
    E_CHARGE, C_LIGHT, EPSILON_0,
    # Conversions
    THZ_TO_CM, CM_TO_THZ, THZ_TO_MEV, MEV_TO_THZ,
    THZ_TO_KELVIN, KELVIN_TO_THZ, MEV_TO_KELVIN,
    BORN_FACTOR,
    # Thermodynamic functions
    bose_einstein, helmholtz_free_energy, phonon_entropy,
    heat_capacity_cv, dulong_petit_limit, zero_point_energy,
    # Stability
    check_dynamical_stability,
    # Hydrogen analytics
    H_MODE_RANGES, H_MODE_RANGES_CM, HYDRIDE_STRETCH_LIBRARY,
    # Elastic constants
    sound_velocity_to_elastic_constant_cubic,
    compute_vrh_moduli, youngs_modulus_from_BG, poisson_ratio_from_BG,
    debye_temperature_from_sound_velocity, check_mechanical_stability_cubic,
    # Anharmonic / Slack model
    slack_thermal_conductivity,
    # H2 molecule properties (from h2_molecule module)
    # Note: H2 functions are in h2_molecule module, not physics
)

# Visualization
from .visualization import (
    PhononFigureFactory,
    BasePlotter,
    THEMES,
    DEFAULT_THEME,
    ELEMENT_COLORS as VIZ_ELEMENT_COLORS,
    get_theme,
    list_themes,
    freq_to_cm,
    freq_to_meV,
)

__all__ = [
    # Core
    'PhononPostProcessor',
    'PostprocessorConfig',
    # Visualization
    'PhononFigureFactory',
    'BasePlotter',
    'THEMES',
    'DEFAULT_THEME',
    'VIZ_ELEMENT_COLORS',
    'get_theme',
    'list_themes',
    'freq_to_cm',
    'freq_to_meV',
]
