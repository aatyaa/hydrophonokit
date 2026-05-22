"""
=============================================================================
  HydroPhonoKit v2.4 -- Equation of State Models

  Implements multiple equations of state (EOS) for fitting E(V) data
  from DFT calculations. Used in quasi-harmonic approximation to extract:
    - E0: equilibrium energy
    - V0: equilibrium volume
    - B0: bulk modulus
    - B0': pressure derivative of bulk modulus

  References:
    [1] Birch, Phys. Rev. 71, 809 (1947) -- Finite strain EOS
    [2] Murnaghan, Proc. Natl. Acad. Sci. 30, 244 (1944)
    [3] Vinet et al., J. Phys. Condens. Matter 1, 1941 (1989)
    [4] Poirier, Introduction to the Physics of the Earth's Interior (2000)
    [5] Lejaeghere et al., Science 351, 1415 (2016) -- Delta project validation
=============================================================================
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional


# ============================================================================
# EQUATION OF STATE MODELS
# ============================================================================

def birch_murnaghan(V, E0, V0, B0, B0_prime):
    """Third-order Birch-Murnaghan Equation of State.

    E(V) = E0 + (9*V0*B0/16) * {[(V0/V)^(2/3) - 1]^3 * B0'
                                 + [(V0/V)^(2/3) - 1]^2 * [6 - 4*(V0/V)^(2/3)]}

    Args:
        V: Volume array (Angstrom^3)
        E0: Equilibrium energy (eV)
        V0: Equilibrium volume (Angstrom^3)
        B0: Bulk modulus (eV/Angstrom^3)
        B0_prime: Pressure derivative of bulk modulus (dimensionless)

    Returns:
        Energy at each volume (eV)

    Reference: Birch (1947), Phys. Rev. 71, 809.
    """
    eta = (V0 / V) ** (2/3)
    f = (eta - 1) / 2  # Eulerian finite strain

    E = E0 + 9 * V0 * B0 / 16 * (
        (eta - 1)**3 * B0_prime +
        (eta - 1)**2 * (6 - 4 * eta)
    )
    return E


def birch_murnaghan_P(V, V0, B0, B0_prime):
    """Pressure from Birch-Murnaghan EOS.

    P(V) = 3*B0/2 * [(V0/V)^(7/3) - (V0/V)^(5/3)]
           * {1 + 3/4*(B0' - 4)*[(V0/V)^(2/3) - 1]}

    Args:
        V: Volume (Angstrom^3)
        V0: Equilibrium volume (Angstrom^3)
        B0: Bulk modulus (eV/Angstrom^3)
        B0_prime: Pressure derivative of bulk modulus

    Returns:
        Pressure (eV/Angstrom^3)
    """
    eta = (V0 / V) ** (2/3)
    P = 3 * B0 / 2 * (eta**(7/2) - eta**(5/2)) * \
        (1 + 3/4 * (B0_prime - 4) * (eta - 1))
    return P


def vinet(V, E0, V0, B0, B0_prime):
    """Vinet Equation of State (Universal EOS).

    E(V) = E0 + 4*B0*V0 / (B0' - 1)^2 *
           {1 - (1 - x)*exp(x)}

    where x = 1.5*(B0' - 1) * [1 - (V/V0)^(1/3)]

    Args:
        V: Volume array (Angstrom^3)
        E0: Equilibrium energy (eV)
        V0: Equilibrium volume (Angstrom^3)
        B0: Bulk modulus (eV/Angstrom^3)
        B0_prime: Pressure derivative of bulk modulus (dimensionless)

    Returns:
        Energy at each volume (eV)

    Reference: Vinet et al. (1989), J. Phys. Condens. Matter 1, 1941.
    """
    if B0_prime <= 1:
        B0_prime = 4.0  # Fallback to prevent division by zero

    x = 1.5 * (B0_prime - 1) * (1.0 - (V / V0) ** (1/3))
    E = E0 + 4.0 * B0 * V0 / (B0_prime - 1)**2 * (
        1.0 - (1.0 - x) * np.exp(x)
    )
    return E


def murnaghan(V, E0, V0, B0, B0_prime):
    """Murnaghan Equation of State.

    E(V) = E0 + B0*V / B0' * [(V0/V)^B0' / (B0' - 1) + 1]
           - V0*B0 / (B0' - 1)

    Args:
        V: Volume array (Angstrom^3)
        E0: Equilibrium energy (eV)
        V0: Equilibrium volume (Angstrom^3)
        B0: Bulk modulus (eV/Angstrom^3)
        B0_prime: Pressure derivative of bulk modulus (dimensionless)

    Returns:
        Energy at each volume (eV)

    Reference: Murnaghan (1944), Proc. Natl. Acad. Sci. 30, 244.
    """
    if B0_prime <= 1:
        B0_prime = 4.0  # Fallback

    E = E0 + B0 * V / B0_prime * (
        (V0 / V)**B0_prime / (B0_prime - 1) + 1
    ) - V0 * B0 / (B0_prime - 1)
    return E


# EOS model registry
EOS_MODELS = {
    'birch_murnaghan': birch_murnaghan,
    'vinet': vinet,
    'murnaghan': murnaghan,
}


# ============================================================================
# EOS FITTING
# ============================================================================

def fit_eos(volumes, energies, model='birch_murnaghan',
            p0=None, bounds=None) -> Dict:
    """Fit equation of state to E(V) data.

    Args:
        volumes: Array of volumes (Angstrom^3)
        energies: Array of energies (eV)
        model: EOS model name ('birch_murnaghan', 'vinet', 'murnaghan')
        p0: Initial guess [E0, V0, B0, B0'] (optional)
        bounds: Parameter bounds for fitting (optional)

    Returns:
        dict: {
            'E0': eV,
            'V0': Angstrom^3,
            'B0': GPa,
            'B0_prime': dimensionless,
            'model': str,
            'r_squared': float,
            'params': [E0, V0, B0, B0'],
            'covariance': array or None
        }

    Note: B0 is converted from eV/A^3 to GPa in output.
    """
    if model not in EOS_MODELS:
        raise ValueError(f"Unknown EOS model: {model}. Choose from {list(EOS_MODELS.keys())}")

    eos_func = EOS_MODELS[model]
    V = np.asarray(volumes, dtype=float)
    E = np.asarray(energies, dtype=float)

    # Initial guess
    if p0 is None:
        E0_guess = np.min(E)
        V0_guess = V[np.argmin(E)]

        # Rough B0 estimate from curvature at minimum
        if len(V) >= 3:
            idx_min = np.argmin(E)
            if idx_min > 0 and idx_min < len(V) - 1:
                d2E_dV2 = (E[idx_min + 1] - 2 * E[idx_min] + E[idx_min - 1]) / \
                          ((V[idx_min + 1] - V[idx_min - 1]) / 2)**2
                B0_guess = V0_guess * d2E_dV2 / 9  # eV/A^3
                B0_guess = max(B0_guess, 0.01)  # Ensure positive
            else:
                B0_guess = 0.1
        else:
            B0_guess = 0.1

        p0 = [E0_guess, V0_guess, B0_guess, 4.0]

    # Default bounds
    if bounds is None:
        bounds = (
            [-np.inf, V0_guess * 0.5, 0.001, 1.0],  # lower
            [np.inf, V0_guess * 1.5, 10.0, 10.0]     # upper
        )

    # Fit
    try:
        popt, pcov = curve_fit(
            eos_func, V, E, p0=p0, bounds=bounds,
            maxfev=10000
        )
    except RuntimeError as e:
        raise RuntimeError(f"EOS fitting failed for {model}: {e}")

    # Convert B0 from eV/A^3 to GPa (1 eV/A^3 = 160.21766208 GPa)
    EV_A3_TO_GPA = 160.21766208
    B0_GPa = popt[2] * EV_A3_TO_GPA

    # Compute R-squared
    E_fit = eos_func(V, *popt)
    SS_res = np.sum((E - E_fit)**2)
    SS_tot = np.sum((E - np.mean(E))**2)
    r_squared = 1 - SS_res / SS_tot if SS_tot > 0 else 0.0

    return {
        'E0': popt[0],           # eV
        'V0': popt[1],           # Angstrom^3
        'B0': B0_GPa,            # GPa
        'B0_prime': popt[3],     # dimensionless
        'model': model,
        'r_squared': r_squared,
        'params': popt.tolist(),
        'covariance': pcov.tolist() if pcov is not None else None,
    }


def fit_all_eos(volumes, energies) -> Dict:
    """Fit all available EOS models and return results.

    Args:
        volumes: Array of volumes (Angstrom^3)
        energies: Array of energies (eV)

    Returns:
        dict: {model_name: fit_result, ...}
    """
    results = {}
    for model in EOS_MODELS:
        try:
            results[model] = fit_eos(volumes, energies, model=model)
        except Exception as e:
            results[model] = {'error': str(e)}
    return results


def best_eos_model(results) -> str:
    """Select best EOS model based on R-squared.

    Args:
        results: dict from fit_all_eos()

    Returns:
        Name of best model
    """
    best_model = None
    best_r2 = -np.inf
    for model, res in results.items():
        if 'error' not in res and res.get('r_squared', -np.inf) > best_r2:
            best_r2 = res['r_squared']
            best_model = model
    return best_model


# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

EV_A3_TO_GPA = 160.21766208
GPA_TO_EV_A3 = 1.0 / EV_A3_TO_GPA


def convert_B0(B0, from_unit='eV/A^3', to_unit='GPa'):
    """Convert bulk modulus between units.

    Args:
        B0: Bulk modulus value
        from_unit: 'eV/A^3' or 'GPa'
        to_unit: 'eV/A^3' or 'GPa'

    Returns:
        Converted bulk modulus
    """
    if from_unit == to_unit:
        return B0

    if from_unit == 'eV/A^3' and to_unit == 'GPa':
        return B0 * EV_A3_TO_GPA
    elif from_unit == 'GPa' and to_unit == 'eV/A^3':
        return B0 * GPA_TO_EV_A3
    else:
        raise ValueError(f"Unknown units: {from_unit} -> {to_unit}")
