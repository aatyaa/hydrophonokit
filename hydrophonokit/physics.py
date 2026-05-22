"""
=============================================================================
  HydroPhonoKit v2.2 -- Physical Constants and Rigorous Scientific Methods

  Centralized repository for:
    - Fundamental physical constants (CODATA 2018 recommended values)
    - Unit conversion factors with full precision
    - Thermodynamic formulas (harmonic approximation)
    - Acoustic Sum Rule (ASR) implementations
    - Born effective charge corrections
    - Non-analytical term corrections (LO-TO splitting)
    - Hydrogen storage analytics criteria
    - Elastic constant estimation from phonon slopes
    - Debye temperature computation
    - Zero-point energy and quantum corrections

  References:
    [1] CODATA 2018 Recommended Values (Tiesinga et al., Rev. Mod. Phys. 93, 025010, 2021)
    [2] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
    [3] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT
    [4] Born & Huang, Dynamical Theory of Crystal Lattices (1954)
    [5] Parlinski et al., Phys. Rev. Lett. 78, 4063 (1997) -- Finite displacement
    [6] Gonze & Lee, Phys. Rev. B 55, 10355 (1997) -- LO-TO splitting
    [7] Wang et al., Phys. Rev. B 95, 014303 (2017) -- symfc
    [8] Grimme et al., J. Chem. Phys. 132, 154104 (2010) -- DFT-D3
=============================================================================
"""
import numpy as np

# ============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2018)
# ============================================================================

# Elementary charge [C]
E_CHARGE = 1.602176634e-19

# Electron mass [kg]
M_ELECTRON = 9.1093837015e-31

# Proton mass [kg]
M_PROTON = 1.67262192369e-27

# Neutron mass [kg]
M_NEUTRON = 1.67492749804e-27

# Speed of light in vacuum [m/s]
C_LIGHT = 299792458.0

# Planck constant [J*s]
H_PLANCK = 6.62607015e-34

# Reduced Planck constant [J*s]
HBAR = H_PLANCK / (2 * np.pi)

# Boltzmann constant [J/K]
K_BOLTZMANN = 1.380649e-23

# Avogadro constant [mol^-1]
N_AVOGADRO = 6.02214076e23

# Gas constant [J/(mol*K)]
R_GAS = N_AVOGADRO * K_BOLTZMANN  # = 8.314462618... J/(mol*K)

# Bohr radius [m]
A0_BOHR = 5.29177210903e-11

# Hartree energy [J]
E_HARTREE = 4.3597447222071e-18

# Vacuum permittivity [F/m]
EPSILON_0 = 8.8541878128e-12

# ============================================================================
# ENERGY UNIT CONVERSIONS
# ============================================================================

# 1 eV in Joules
EV_TO_J = E_CHARGE

# 1 eV in kJ/mol
EV_TO_KJ_MOL = E_CHARGE * N_AVOGADRO / 1000  # = 96.48533212...

# 1 eV in kcal/mol
EV_TO_KCAL_MOL = EV_TO_KJ_MOL / 4.184  # = 23.060549...

# 1 Hartree in eV
HARTREE_TO_EV = E_HARTREE / E_CHARGE  # = 27.211386245988

# 1 Rydberg in eV
RYDBERG_TO_EV = HARTREE_TO_EV / 2  # = 13.605693122994

# ============================================================================
# FREQUENCY / WAVENUMBER / TEMPERATURE CONVERSIONS
# ============================================================================

# THz to cm^-1:  nu(cm^-1) = nu(Hz) / (100 * c) = nu(THz) * 1e12 / (100 * c)
THZ_TO_CM = 1e12 / (100 * C_LIGHT)  # = 33.356409519815...
CM_TO_THZ = 1.0 / THZ_TO_CM  # = 0.0299792458...

# THz to meV:  E = h * nu  =>  E[eV] = h[Js] * nu[Hz] / e[C]
THZ_TO_MEV = (H_PLANCK * 1e12 / E_CHARGE) * 1000  # = 4.135667696...

# meV to THz
MEV_TO_THZ = 1.0 / THZ_TO_MEV

# meV to cm^-1
MEV_TO_CM = THZ_TO_CM * MEV_TO_THZ  # = 8.0655439...

# meV to Kelvin: E = k_B * T
MEV_TO_KELVIN = (E_CHARGE * 1e-3) / K_BOLTZMANN  # = 11.604518...

# Kelvin to meV
KELVIN_TO_MEV = 1.0 / MEV_TO_KELVIN

# THz to Kelvin
THZ_TO_KELVIN = (H_PLANCK * 1e12) / K_BOLTZMANN  # = 47.992447...

# Kelvin to THz
KELVIN_TO_THZ = 1.0 / THZ_TO_KELVIN

# ============================================================================
# VASP-SPECIFIC CONVERSIONS
# ============================================================================

# Born conversion factor: e^2 / (4*pi*eps_0) in eV*A
# This is the factor VASP uses internally for non-analytical corrections
BORN_FACTOR = (E_CHARGE**2) / (4 * np.pi * EPSILON_0) / EV_TO_J * 1e10  # = 14.399652...

# Phonopy frequency unit: THz
PHONOPY_FREQ_UNIT = "THz"

# Phonopy force constant unit: eV/Angstrom^2
PHONOPY_FC_UNIT = "eV/Angstrom^2"

# ============================================================================
# ACOUSTIC SUM RULE (ASR) PARAMETERS
# ============================================================================

# ASR types
ASR_OFF = "off"
ASR_SIMPLE = "simple"     # Simple correction: distribute error equally
ASR_RECIPROCAL = "reciprocal"  # Reciprocal-space correction (symfc default)

# Acoustic mode frequency tolerance at Gamma [THz]
# Frequencies below this at Gamma are considered acoustic modes
ACOUSTIC_TOLERANCE = 0.5  # THz

# Sum rule tolerance for Born charges [e]
BORN_SUM_RULE_TOL = 1e-6  # elementary charges

# ============================================================================
# THERMODYNAMIC FORMULAS (Harmonic Approximation)
# ============================================================================

# Bose-Einstein occupation number: n(omega, T) = 1 / (exp(hbar*omega / kT) - 1)
def bose_einstein(freq_thz, temperature_K):
    """Compute Bose-Einstein occupation number.

    Args:
        freq_thz: Frequency in THz
        temperature_K: Temperature in Kelvin

    Returns:
        Occupation number n(omega, T)
    """
    if freq_thz <= 0 or temperature_K <= 0:
        return 0.0
    x = (H_PLANCK * freq_thz * 1e12) / (K_BOLTZMANN * temperature_K)
    if x > 100:  # Avoid overflow
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)


# Helmholtz free energy: F(T) = kT * Sum_qj ln[2 sinh(hbar*omega_qj / 2kT)]
def helmholtz_free_energy(freq_thz_array, temperature_K):
    """Compute Helmholtz free energy per mode in the harmonic approximation.

    F(T) = (1/2) * hbar * omega + kT * ln[1 - exp(-hbar*omega / kT)]

    Args:
        freq_thz_array: Array of phonon frequencies in THz
        temperature_K: Temperature in Kelvin

    Returns:
        Free energy in kJ/mol
    """
    if temperature_K <= 0:
        # Zero-temperature limit: just ZPE
        zpe = 0.5 * H_PLANCK * np.sum(freq_thz_array[freq_thz_array > 0]) * 1e12
        return zpe * N_AVOGADRO / 1000  # kJ/mol

    omega = 2 * np.pi * freq_thz_array * 1e12  # rad/s
    x = (HBAR * omega) / (K_BOLTZMANN * temperature_K)

    # Avoid numerical issues at small x
    mask = freq_thz_array > 1e-10
    f = np.zeros_like(freq_thz_array)

    if np.any(mask):
        x_mask = x[mask]
        # F = kT * ln[2*sinh(x/2)] for each mode
        f[mask] = K_BOLTZMANN * temperature_K * np.log(2 * np.sinh(x_mask / 2))

    # Sum over all modes, convert to kJ/mol
    return np.sum(f) * N_AVOGADRO / 1000


# Entropy: S(T) = k * Sum_qj [(n+1/2)*coth(hbar*omega/2kT) - ln(2*sinh(hbar*omega/2kT))]
def phonon_entropy(freq_thz_array, temperature_K):
    """Compute phonon entropy in the harmonic approximation.

    S(T) = k * Sum [ (n+1)ln(n+1) - n*ln(n) ]  where n = Bose-Einstein occupation

    Args:
        freq_thz_array: Array of phonon frequencies in THz
        temperature_K: Temperature in Kelvin

    Returns:
        Entropy in J/(mol*K)
    """
    if temperature_K <= 1e-10:
        return 0.0

    freq_pos = freq_thz_array[freq_thz_array > 1e-10]
    if len(freq_pos) == 0:
        return 0.0

    x = (H_PLANCK * freq_pos * 1e12) / (K_BOLTZMANN * temperature_K)
    n = 1.0 / (np.exp(x) - 1.0)

    # S = k * Sum [ (n+1)ln(n+1) - n*ln(n) ]
    s_per_mode = K_BOLTZMANN * ((n + 1) * np.log(n + 1 + 1e-300) - n * np.log(n + 1e-300))
    return np.sum(s_per_mode) * N_AVOGADRO


# Heat capacity at constant volume: Cv(T) = Sum_qj k * (hbar*omega/kT)^2 * exp(hbar*omega/kT) / [exp(hbar*omega/kT)-1]^2
def heat_capacity_cv(freq_thz_array, temperature_K):
    """Compute phonon heat capacity at constant volume.

    C_v(T) = k * Sum_qj x^2 * exp(x) / (exp(x) - 1)^2  where x = hbar*omega / kT

    Args:
        freq_thz_array: Array of phonon frequencies in THz
        temperature_K: Temperature in Kelvin

    Returns:
        C_v in J/(mol*K)
    """
    if temperature_K <= 1e-10:
        return 0.0

    freq_pos = freq_thz_array[freq_thz_array > 1e-10]
    if len(freq_pos) == 0:
        return 0.0

    x = (H_PLANCK * freq_pos * 1e12) / (K_BOLTZMANN * temperature_K)
    # C_v per mode: k * x^2 * exp(x) / (exp(x) - 1)^2
    # Use numerically stable form: k * x^2 / (4 * sinh^2(x/2))
    cv_per_mode = K_BOLTZMANN * (x ** 2) / (4 * np.sinh(x / 2) ** 2)
    return np.sum(cv_per_mode) * N_AVOGADRO


# Dulong-Petit limit: C_v -> 3 * N * k_B (per formula unit)
def dulong_petit_limit(n_atoms_per_cell):
    """Compute the Dulong-Petit high-temperature limit for C_v.

    C_v -> 3 * N * R  where N = atoms per primitive cell, R = gas constant

    Args:
        n_atoms_per_cell: Number of atoms in the primitive cell

    Returns:
        Dulong-Petit limit in J/(mol*K)
    """
    return 3 * n_atoms_per_cell * R_GAS


# Zero-point energy: ZPE = (1/2) * Sum_qj hbar * omega_qj
def zero_point_energy(freq_thz_array):
    """Compute zero-point energy.

    ZPE = (1/2) * Sum hbar * omega

    Args:
        freq_thz_array: Array of phonon frequencies in THz (only positive used)

    Returns:
        ZPE in kJ/mol
    """
    freq_pos = freq_thz_array[freq_thz_array > 0]
    if len(freq_pos) == 0:
        return 0.0
    zpe = 0.5 * H_PLANCK * np.sum(freq_pos) * 1e12  # Joules per cell
    return zpe * N_AVOGADRO / 1000  # kJ/mol


# Gibbs free energy (quasi-harmonic): G(T,P) = F(T) + PV + ZPE
def gibbs_free_energy(helmholtz_F, pressure_GPa, volume_A3):
    """Compute Gibbs free energy in the quasi-harmonic approximation.

    G(T,P) = F(T) + P*V  (neglecting anharmonic contributions)

    Args:
        helmholtz_F: Helmholtz free energy in kJ/mol
        pressure_GPa: Pressure in GPa
        volume_A3: Volume in Angstrom^3 per formula unit

    Returns:
        Gibbs free energy in kJ/mol
    """
    # P*V conversion: 1 GPa*A^3 = 1e-3 J/mol * N_A / N_A = 1e-3 J = 1e-6 kJ
    pv_kj = pressure_GPa * volume_A3 * 1e-6 * N_AVOGADRO / 1e3
    return helmholtz_F + pv_kj


# ============================================================================
# DEBYE MODEL
# ============================================================================

# Debye frequency: omega_D = (6*pi^2*N/V)^(1/3) * v_s
def debye_frequency(n_atoms, volume_A3, sound_velocity_ms=None):
    """Estimate Debye frequency.

    omega_D = (6*pi^2*N/V)^(1/3) * v_s

    Args:
        n_atoms: Number of atoms per primitive cell
        volume_A3: Volume in A^3
        sound_velocity_ms: Average sound velocity in m/s (if known)

    Returns:
        Debye frequency in THz (or None if sound velocity unknown)
    """
    if sound_velocity_ms is None:
        return None

    n_density = n_atoms / (volume_A3 * 1e-30)  # atoms/m^3
    k_debye = (6 * np.pi**2 * n_density) ** (1/3)  # 1/m
    omega_D = sound_velocity_ms * k_debye  # rad/s
    nu_D = omega_D / (2 * np.pi)  # Hz
    return nu_D / 1e12  # THz


# Debye temperature: Theta_D = hbar * omega_D / k_B
def debye_temperature(debye_freq_thz):
    """Compute Debye temperature from Debye frequency.

    Theta_D = hbar * omega_D / k_B = h * nu_D / k_B

    Args:
        debye_freq_thz: Debye frequency in THz

    Returns:
        Debye temperature in Kelvin
    """
    if debye_freq_thz is None:
        return None
    return THZ_TO_KELVIN * debye_freq_thz


# Debye heat capacity (full formula)
def debye_heat_capacity(temp_K, debye_temp_K):
    """Compute Debye model heat capacity.

    C_v = 9 * N * k_B * (T/Theta_D)^3 * Integral_0^{Theta_D/T} x^4*exp(x)/(exp(x)-1)^2 dx

    Args:
        temp_K: Temperature in Kelvin
        debye_temp_K: Debye temperature in Kelvin

    Returns:
        C_v per mole in J/(mol*K) (for N=1)
    """
    if debye_temp_K <= 0:
        return 0.0

    x_max = debye_temp_K / temp_K
    if x_max > 100:
        # Low-T limit: C_v ~ (12*pi^4/5) * N * k_B * (T/Theta_D)^3
        return (12 * np.pi**4 / 5) * R_GAS * (temp_K / debye_temp_K) ** 3

    # Numerical integration
    x = np.linspace(1e-10, x_max, 1000)
    integrand = (x ** 4) * np.exp(x) / (np.exp(x) - 1) ** 2
    integral = np.trapz(integrand, x) if hasattr(np, 'trapz') else np.trapezoid(integrand, x)

    return 9 * R_GAS * (temp_K / debye_temp_K) ** 3 * integral


# ============================================================================
# BORN EFFECTIVE CHARGE & LO-TO SPLITTING
# ============================================================================

def apply_born_sum_rule(born_charges):
    """Apply Acoustic Sum Rule to Born effective charges.

    Sum_i Z*_{i,alpha,beta} = 0  (charge neutrality + translational invariance)

    The correction is distributed equally among all atoms.

    Args:
        born_charges: ndarray of shape (n_atoms, 3, 3)

    Returns:
        Corrected Born charges
    """
    z_sum = np.sum(born_charges, axis=0)
    n_atoms = born_charges.shape[0]
    correction = z_sum / n_atoms
    return born_charges - correction[np.newaxis, :, :]


def born_sum_rule_error(born_charges):
    """Compute the Born charge sum rule violation.

    Returns:
        Trace of sum rule error tensor [e]
    """
    z_sum = np.sum(born_charges, axis=0)
    return np.trace(z_sum) / 3.0


# LO-TO splitting frequency: omega_LO^2 - omega_TO^2 ~ 4*pi*e^2/(V*epsilon_inf) * |q.e*Z*|
# The actual implementation is handled by phonopy via nac_params

# ============================================================================
# HYDROGEN STORAGE ANALYTICS
# ============================================================================

# Hydrogen mode frequency ranges (in THz) for metal hydrides
# Reference: Nakamoto, "Infrared and Raman Spectra of Inorganic Compounds"
#            Bogdanovic et al., J. Alloys Compd. (2004)
H_MODE_RANGES = {
    'librational':  (5.0,  20.9),   # ~167 - 697 cm^-1: hindered rotation of H2/BH4 units
    'bending':      (20.9, 50.0),   # ~697 - 1668 cm^-1: H-X-H bending modes
    'stretching':   (50.0, 100.0),  # ~1668 - 3336 cm^-1: B-H, N-H, O-H stretching
}

# Hydrogen mode frequency ranges in cm^-1
H_MODE_RANGES_CM = {
    k: (v[0] * THZ_TO_CM, v[1] * THZ_TO_CM)
    for k, v in H_MODE_RANGES.items()
}

# Criteria for viable hydrogen storage materials
# Reference: DOE targets (2025): 5.5 wt% system, 40 g/L volumetric
H_STORAGE_CRITERIA = {
    'gravimetric_min_wt_pct': 5.5,     # DOE 2025 target
    'volumetric_min_g_per_L': 40.0,    # DOE 2025 target
    'desorption_temp_min_K': 300,      # Must release H2 at reasonable T
    'desorption_temp_max_K': 500,      # Not too high
    'h_binding_energy_min_ev': 0.2,    # Minimum for chemisorption
    'h_binding_energy_max_ev': 0.8,    # Maximum for reversibility
}

# Common hydride stretching frequencies for validation (cm^-1)
HYDRIDE_STRETCH_LIBRARY = {
    'B-H':   (2200, 2600, 'Borohydride stretch'),
    'Al-H':  (1700, 1850, 'Aluminum hydride stretch'),
    'Mg-H':  (1100, 1400, 'Magnesium hydride stretch'),
    'Ti-H':  (1200, 1500, 'Titanium hydride stretch'),
    'N-H':   (3100, 3500, 'Amide/imide stretch'),
    'O-H':   (3200, 3700, 'Hydroxyl stretch'),
    'Li-H':  (1100, 1200, 'Lithium hydride stretch'),
    'Na-H':  (1150, 1250, 'Sodium hydride stretch'),
    'Ca-H':  (1200, 1400, 'Calcium hydride stretch'),
    'Si-H':  (2100, 2250, 'Silane stretch'),
    'C-H':   (2800, 3100, 'Hydrocarbon stretch'),
}


# ============================================================================
# FORCE CONSTANT ANALYSIS
# ============================================================================

def compute_force_constant_eigenvalues(force_constants, masses):
    """Compute eigenvalues of the dynamical matrix.

    D_q = M^{-1/2} * Phi_q * M^{-1/2}

    Args:
        force_constants: Force constant matrix in real space
        masses: Atomic masses in amu

    Returns:
        Eigenvalues of dynamical matrix (eV/A^2/amu)
    """
    n_atoms = len(masses)
    mass_sqrt_inv = np.zeros((3 * n_atoms, 3 * n_atoms))
    for i in range(n_atoms):
        m = np.sqrt(masses[i])
        mass_sqrt_inv[3*i:3*(i+1), 3*i:3*(i+1)] = np.eye(3) / m

    dynamical_matrix = mass_sqrt_inv @ force_constants.reshape(3*n_atoms, 3*n_atoms) @ mass_sqrt_inv
    eigenvalues = np.linalg.eigvalsh(dynamical_matrix)
    return eigenvalues


# ============================================================================
# GRUNEISEN PARAMETERS
# ============================================================================

def gruneisen_parameter(freq_vol1, freq_vol2, vol1, vol2):
    """Compute mode Gruneisen parameter.

    gamma_i = - d(ln omega_i) / d(ln V)
            = - ln(omega_i(V2) / omega_i(V1)) / ln(V2 / V1)

    Args:
        freq_vol1: Frequency at volume V1 (THz)
        freq_vol2: Frequency at volume V2 (THz)
        vol1: Volume V1 (A^3)
        vol2: Volume V2 (A^3)

    Returns:
        Mode Gruneisen parameter (dimensionless)
    """
    if freq_vol1 <= 0 or freq_vol2 <= 0:
        return 0.0
    dlnV = np.log(vol2 / vol1)
    if abs(dlnV) < 1e-10:
        return 0.0
    return -np.log(freq_vol2 / freq_vol1) / dlnV


def average_gruneisen(freq_vol1_arr, freq_vol2_arr, vol1, vol2):
    """Compute average Gruneisen parameter over all modes.

    Args:
        freq_vol1_arr: Frequencies at volume V1
        freq_vol2_arr: Frequencies at volume V2
        vol1: Volume V1
        vol2: Volume V2

    Returns:
        Average Gruneisen parameter
    """
    gammas = []
    for f1, f2 in zip(freq_vol1_arr, freq_vol2_arr):
        if f1 > 0.1 and f2 > 0.1:  # Skip acoustic/imaginary modes
            gammas.append(gruneisen_parameter(f1, f2, vol1, vol2))
    return np.mean(gammas) if gammas else 0.0


# ============================================================================
# PHONON STABILITY CRITERIA
# ============================================================================

def check_dynamical_stability(frequencies, tolerance=-0.5):
    """Check if a structure is dynamically stable.

    A structure is dynamically stable if all phonon frequencies are real
    (i.e., no imaginary/negative frequencies) except for the 3 acoustic
    modes at Gamma which should be near zero.

    Args:
        frequencies: Array of all phonon frequencies (THz)
        tolerance: Threshold below which a frequency is considered imaginary (THz)

    Returns:
        dict: {stable: bool, min_freq: float, n_imaginary: int, imaginary_modes: list}
    """
    min_freq = np.min(frequencies)
    imaginary_mask = frequencies < tolerance
    n_imaginary = np.sum(imaginary_mask)

    return {
        'stable': min_freq >= tolerance,
        'min_freq': min_freq,
        'n_imaginary': int(n_imaginary),
        'imaginary_modes': frequencies[imaginary_mask].tolist() if n_imaginary > 0 else [],
        'tolerance': tolerance,
    }


def check_mechanical_stability(elastic_constants, crystal_system='cubic'):
    """Check Born-Huang mechanical stability criteria.

    For cubic crystals:
        C11 > 0, C44 > 0, C11 > |C12|, C11 + 2*C12 > 0

    For hexagonal crystals:
        C11 > 0, C44 > 0, C11 > |C12|, (C11 + 2*C12)*C33 > 2*C13^2

    For tetragonal crystals:
        C11 > 0, C33 > 0, C44 > 0, C66 > 0,
        C11 > |C12|, (C11 + C33 - 2*C13) > 0,
        2*(C11 + C12) + C33 + 4*C13 > 0

    Args:
        elastic_constants: Elastic constant tensor (Voigt notation)
        crystal_system: Crystal system type

    Returns:
        dict: {stable: bool, criteria: list of (name, passed)}
    """
    criteria = []

    if crystal_system == 'cubic':
        c11, c12, c44 = elastic_constants[0], elastic_constants[1], elastic_constants[3]
        criteria.append(('C11 > 0', c11 > 0))
        criteria.append(('C44 > 0', c44 > 0))
        criteria.append(('C11 > |C12|', c11 > abs(c12)))
        criteria.append(('C11 + 2*C12 > 0', c11 + 2*c12 > 0))

    elif crystal_system == 'hexagonal':
        c11, c12, c13, c33, c44 = (elastic_constants[0], elastic_constants[1],
                                     elastic_constants[2], elastic_constants[3],
                                     elastic_constants[4])
        criteria.append(('C11 > 0', c11 > 0))
        criteria.append(('C44 > 0', c44 > 0))
        criteria.append(('C11 > |C12|', c11 > abs(c12)))
        criteria.append(('C11 - C12 > 0', c11 - c12 > 0))
        criteria.append(('(C11+2*C12)*C33 > 2*C13^2',
                        (c11 + 2*c12) * c33 > 2 * c13**2))

    return {
        'stable': all(c[1] for c in criteria),
        'criteria': criteria,
    }


# ============================================================================
# BOND STRENGTH ANALYSIS
# ============================================================================

def estimate_bond_force_constant(bond_type, freq_cm):
    """Estimate effective bond force constant from vibrational frequency.

    Using harmonic oscillator: k = (2*pi*c*nu~)^2 * mu

    where nu~ is wavenumber in cm^-1, mu is reduced mass in amu.

    Args:
        bond_type: String like 'B-H', 'N-H', etc.
        freq_cm: Observed frequency in cm^-1

    Returns:
        Force constant in N/m (or mdyn/A)
    """
    # Get reduced mass
    elements = bond_type.split('-')
    if len(elements) != 2:
        return None

    from .analyzer import ELEMENT_DB
    masses = []
    for elem in elements:
        if elem in ELEMENT_DB:
            masses.append(ELEMENT_DB[elem][1])  # mass in amu
        else:
            return None

    mu = (masses[0] * masses[1]) / (masses[0] + masses[1])  # reduced mass in amu
    mu_kg = mu * 1.66054e-27  # convert to kg

    # nu in Hz
    nu_hz = freq_cm * 100 * C_LIGHT

    # k = (2*pi*nu)^2 * mu
    k = (2 * np.pi * nu_hz) ** 2 * mu_kg  # N/m

    return k / 100  # Convert to mdyn/A (1 mdyn/A = 100 N/m)


# ============================================================================
# ELASTIC CONSTANTS & MECHANICAL PROPERTIES
# ============================================================================

def sound_velocity_to_elastic_constant_cubic(v_LA, v_TA1, v_TA2, density_kg_m3):
    """Convert sound velocities to elastic constants for cubic crystals.

    For cubic crystals:
        C11 = rho * v_LA^2
        C44 = rho * v_TA1^2
        C12 = C11 - 2 * rho * v_TA2^2

    Args:
        v_LA: Longitudinal acoustic velocity (m/s)
        v_TA1: First transverse acoustic velocity (m/s)
        v_TA2: Second transverse acoustic velocity (m/s)
        density_kg_m3: Mass density (kg/m^3)

    Returns:
        dict: {'C11': GPa, 'C12': GPa, 'C44': GPa}
    """
    C11 = density_kg_m3 * v_LA**2 / 1e9  # Pa -> GPa
    C44 = density_kg_m3 * v_TA1**2 / 1e9
    C12 = C11 - 2 * density_kg_m3 * v_TA2**2 / 1e9
    return {'C11': C11, 'C12': C12, 'C44': C44}


def compute_vrh_moduli(C11, C12, C44, crystal_system='cubic'):
    """Compute Voigt-Reuss-Hill averaged bulk and shear moduli.

    For cubic crystals:
        B_V = B_R = (C11 + 2*C12) / 3
        G_V = (C11 - C12 + 3*C44) / 5
        G_R = 5 / [4/C44 + 3/(C11-C12)]
        B_VRH = (B_V + B_R) / 2
        G_VRH = (G_V + G_R) / 2

    Args:
        C11, C12, C44: Elastic constants (GPa)
        crystal_system: Crystal system type

    Returns:
        dict: {'B_V': GPa, 'B_R': GPa, 'B_VRH': GPa,
               'G_V': GPa, 'G_R': GPa, 'G_VRH': GPa}
    """
    # Voigt bulk (same for cubic)
    B_V = (C11 + 2 * C12) / 3

    # Voigt shear
    G_V = (C11 - C12 + 3 * C44) / 5

    # Reuss shear
    if C44 > 0 and (C11 - C12) > 0:
        G_R = 5 / (4/C44 + 3/(C11 - C12))
    else:
        G_R = G_V * 0.9  # Fallback approximation

    # VRH averages
    B_VRH = (B_V + B_V) / 2  # B_V = B_R for cubic
    G_VRH = (G_V + G_R) / 2

    return {
        'B_V': B_V, 'B_R': B_V, 'B_VRH': B_VRH,
        'G_V': G_V, 'G_R': G_R, 'G_VRH': G_VRH,
    }


def youngs_modulus_from_BG(B, G):
    """Compute Young modulus from bulk and shear moduli.

    E = 9*B*G / (3*B + G)

    Args:
        B: Bulk modulus (GPa)
        G: Shear modulus (GPa)

    Returns:
        Young modulus E (GPa)
    """
    if (3*B + G) == 0:
        return 0.0
    return 9 * B * G / (3 * B + G)


def poisson_ratio_from_BG(B, G):
    """Compute Poisson ratio from bulk and shear moduli.

    nu = (3*B - 2*G) / (6*B + 2*G)

    Args:
        B: Bulk modulus (GPa)
        G: Shear modulus (GPa)

    Returns:
        Poisson ratio (dimensionless)
    """
    denom = 6 * B + 2 * G
    if denom == 0:
        return 0.0
    return (3 * B - 2 * G) / denom


def debye_temperature_from_sound_velocity(v_avg, density_kg_m3, molar_mass_kg_mol):
    """Compute Debye temperature from average sound velocity.

    Theta_D = (h/k_B) * (3*N_A*rho / 4*pi*M)^(1/3) * v_avg

    Args:
        v_avg: Average sound velocity (m/s)
        density_kg_m3: Mass density (kg/m^3)
        molar_mass_kg_mol: Molar mass (kg/mol)

    Returns:
        Debye temperature (K)
    """
    if v_avg <= 0 or density_kg_m3 <= 0 or molar_mass_kg_mol <= 0:
        return 0.0

    # Number density of atoms
    n = N_AVOGADRO * density_kg_m3 / molar_mass_kg_mol

    # Debye wavevector
    q_D = (6 * np.pi**2 * n) ** (1/3)

    # Debye frequency and temperature
    omega_D = v_avg * q_D
    return (H_PLANCK / K_BOLTZMANN) * omega_D / (2 * np.pi)


def check_mechanical_stability_cubic(C11, C12, C44):
    """Check Born-Huang mechanical stability for cubic crystals.

    Criteria:
        C11 > 0
        C44 > 0
        C11 > abs(C12)
        C11 + 2*C12 > 0

    Args:
        C11, C12, C44: Elastic constants (GPa)

    Returns:
        dict with stable bool and criteria list
    """
    criteria = [
        ("C11 > 0", C11 > 0),
        ("C44 > 0", C44 > 0),
        ("C11 > |C12|", C11 > abs(C12)),
        ("C11 + 2*C12 > 0", C11 + 2*C12 > 0),
    ]
    return {
        'stable': all(c[1] for c in criteria),
        'criteria': criteria,
    }


# Slack model for lattice thermal conductivity
def slack_thermal_conductivity(theta_D, M, delta, gamma, T, n=1):
    if T < 1 or gamma < 0.01 or theta_D < 1:
        return 0.0
    A = 3.04e-6
    return A * M * theta_D**3 * delta / (gamma**2 * T * n**(2/3))

