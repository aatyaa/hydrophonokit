"""
=============================================================================
  HydroPhonoKit Visualization — Publication-Ready Themes

  Provides scientifically vetted themes for journal publication:
    - Nature: Nature family journals (serif, muted colors, 300 DPI)
    - Science: Science journal (bold sans-serif, high contrast)
    - PRL: Physical Review Letters (B&W friendly, minimal)
    - ACS: American Chemical Society (specific color palette)
    - Presentation: Dark background, bright colors for slides
    - Interactive: Plotly-compatible for web exploration

  References:
    [1] Nature Publishing Guide -- Figure preparation standards
    [2] Science Journal Guidelines -- Color and formatting
    [3] APS Style Guide -- PRL figure requirements
    [4] ACS Publications Guide -- Color palette specifications
=============================================================================
"""
import os
from typing import Dict, Any

# Configure matplotlib backend
import os as _os
if not _os.environ.get('MPLBACKEND'):
    import matplotlib
    matplotlib.use('Agg')
del _os


# ============================================================================
# THEME DEFINITIONS
# ============================================================================

THEMES: Dict[str, Dict[str, Any]] = {
    'nature': {
        'description': 'Nature family journals - serif, muted colors, 300 DPI',
        'font': {'family': 'serif', 'size': 11, 'weight': 'normal'},
        'figure': {'dpi': 300, 'facecolor': 'white', 'edgecolor': 'white'},
        'axes': {
            'linewidth': 1.0,
            'labelsize': 12,
            'titlesize': 13,
            'grid_alpha': 0.15,
            'labelcolor': '#1e293b',
            'titlecolor': '#1e293b',
        },
        'colors': {
            'primary': '#1E3A8A',      # Deep blue
            'secondary': '#DC2626',    # Red
            'tertiary': '#059669',     # Green
            'quaternary': '#D97706',   # Amber
            'band': '#1E3A8A',
            'band_imaginary': '#DC2626',
            'dos_fill': '#1E3A8A',
            'dos_line': '#1E3A8A',
            'zero_line': '#DC2626',
            'grid': '#94A3B8',
        },
        'legend': {'fontsize': 10, 'framealpha': 0.9, 'loc': 'best'},
        'savefig': {'dpi': 300, 'format': 'png', 'bbox_inches': 'tight'},
        'lines': {'linewidth': 1.5, 'markersize': 6},
    },

    'science': {
        'description': 'Science journal - bold sans-serif, high contrast',
        'font': {'family': 'sans-serif', 'size': 12, 'weight': 'bold'},
        'figure': {'dpi': 300, 'facecolor': 'white', 'edgecolor': 'white'},
        'axes': {
            'linewidth': 1.5,
            'labelsize': 13,
            'titlesize': 14,
            'grid_alpha': 0.2,
            'labelcolor': '#000000',
            'titlecolor': '#000000',
        },
        'colors': {
            'primary': '#000000',      # Black
            'secondary': '#FF0000',    # Bright red
            'tertiary': '#0066CC',     # Blue
            'quaternary': '#009900',   # Green
            'band': '#000000',
            'band_imaginary': '#FF0000',
            'dos_fill': '#000000',
            'dos_line': '#000000',
            'zero_line': '#FF0000',
            'grid': '#CCCCCC',
        },
        'legend': {'fontsize': 11, 'framealpha': 0.95, 'loc': 'best'},
        'savefig': {'dpi': 300, 'format': 'pdf', 'bbox_inches': 'tight'},
        'lines': {'linewidth': 2.0, 'markersize': 7},
    },

    'prl': {
        'description': 'Physical Review Letters - B&W friendly, minimal',
        'font': {'family': 'serif', 'size': 10, 'weight': 'normal'},
        'figure': {'dpi': 600, 'facecolor': 'white', 'edgecolor': 'white'},
        'axes': {
            'linewidth': 0.8,
            'labelsize': 11,
            'titlesize': 12,
            'grid_alpha': 0.0,
            'labelcolor': '#000000',
            'titlecolor': '#000000',
        },
        'colors': {
            'primary': '#000000',      # Black
            'secondary': '#666666',    # Gray
            'tertiary': '#999999',     # Light gray
            'quaternary': '#333333',   # Dark gray
            'band': '#000000',
            'band_imaginary': '#666666',
            'dos_fill': '#000000',
            'dos_line': '#000000',
            'zero_line': '#666666',
            'grid': '#FFFFFF',
        },
        'legend': {'fontsize': 9, 'framealpha': 0.0, 'loc': 'best'},
        'savefig': {'dpi': 600, 'format': 'eps', 'bbox_inches': 'tight'},
        'lines': {'linewidth': 1.2, 'markersize': 5},
    },

    'acs': {
        'description': 'American Chemical Society - specific color palette',
        'font': {'family': 'sans-serif', 'size': 11, 'weight': 'normal'},
        'figure': {'dpi': 300, 'facecolor': 'white', 'edgecolor': 'white'},
        'axes': {
            'linewidth': 1.2,
            'labelsize': 12,
            'titlesize': 13,
            'grid_alpha': 0.15,
            'labelcolor': '#1e293b',
            'titlecolor': '#1e293b',
        },
        'colors': {
            'primary': '#003366',      # ACS Blue
            'secondary': '#CC0000',    # ACS Red
            'tertiary': '#006633',     # ACS Green
            'quaternary': '#FF6600',   # ACS Orange
            'band': '#003366',
            'band_imaginary': '#CC0000',
            'dos_fill': '#003366',
            'dos_line': '#003366',
            'zero_line': '#CC0000',
            'grid': '#94A3B8',
        },
        'legend': {'fontsize': 10, 'framealpha': 0.9, 'loc': 'best'},
        'savefig': {'dpi': 300, 'format': 'tiff', 'bbox_inches': 'tight'},
        'lines': {'linewidth': 1.5, 'markersize': 6},
    },

    'presentation': {
        'description': 'Dark background, bright colors for presentations',
        'font': {'family': 'sans-serif', 'size': 14, 'weight': 'normal'},
        'figure': {
            'dpi': 150,
            'facecolor': '#1a1a2e',
            'edgecolor': '#1a1a2e',
        },
        'axes': {
            'linewidth': 1.5,
            'labelsize': 14,
            'titlesize': 16,
            'grid_alpha': 0.1,
            'labelcolor': '#eaeaea',
            'titlecolor': '#ffffff',
            'facecolor': '#16213e',
        },
        'colors': {
            'primary': '#60A5FA',      # Bright blue
            'secondary': '#F87171',    # Bright red
            'tertiary': '#34D399',     # Bright green
            'quaternary': '#FBBF24',   # Bright yellow
            'band': '#60A5FA',
            'band_imaginary': '#F87171',
            'dos_fill': '#60A5FA',
            'dos_line': '#60A5FA',
            'zero_line': '#F87171',
            'grid': '#333333',
        },
        'legend': {
            'fontsize': 12,
            'framealpha': 0.8,
            'loc': 'best',
            'facecolor': '#16213e',
            'edgecolor': '#333333',
        },
        'savefig': {
            'dpi': 150,
            'format': 'png',
            'bbox_inches': 'tight',
            'facecolor': '#1a1a2e',
            'edgecolor': '#1a1a2e',
        },
        'lines': {'linewidth': 2.0, 'markersize': 8},
    },

    'minimal': {
        'description': 'Minimalist - black & white, no grid, EPS output',
        'font': {'family': 'serif', 'size': 10, 'weight': 'normal'},
        'figure': {'dpi': 600, 'facecolor': 'white', 'edgecolor': 'white'},
        'axes': {
            'linewidth': 0.8,
            'labelsize': 10,
            'titlesize': 11,
            'grid_alpha': 0.0,
            'labelcolor': '#000000',
            'titlecolor': '#000000',
        },
        'colors': {
            'primary': '#000000',
            'secondary': '#404040',
            'tertiary': '#808080',
            'quaternary': '#202020',
            'band': '#000000',
            'band_imaginary': '#404040',
            'dos_fill': '#000000',
            'dos_line': '#000000',
            'zero_line': '#404040',
            'grid': '#FFFFFF',
        },
        'legend': {'fontsize': 9, 'framealpha': 0.0, 'loc': 'best'},
        'savefig': {'dpi': 600, 'format': 'eps', 'bbox_inches': 'tight'},
        'lines': {'linewidth': 1.0, 'markersize': 4},
    },
}

# Default theme
DEFAULT_THEME = 'nature'

# Element color palette (extended from previous versions)
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


# ============================================================================
# THEME MANAGEMENT
# ============================================================================

def get_theme(name: str = DEFAULT_THEME) -> Dict[str, Any]:
    """Get theme configuration by name.
    
    Args:
        name: Theme name ('nature', 'science', 'prl', 'acs', 'presentation', 'minimal')
    
    Returns:
        Theme configuration dictionary
    
    Raises:
        ValueError: If theme name is not recognized
    """
    if name not in THEMES:
        raise ValueError(
            f"Unknown theme: '{name}'. "
            f"Available themes: {list(THEMES.keys())}"
        )
    return THEMES[name]


def list_themes() -> list:
    """List available theme names and descriptions.
    
    Returns:
        List of (name, description) tuples
    """
    return [(name, theme['description']) for name, theme in THEMES.items()]


def get_element_color(element: str) -> str:
    """Get color for an element symbol.
    
    Args:
        element: Element symbol (e.g., 'H', 'Mg', 'B')
    
    Returns:
        Hex color code
    """
    return ELEMENT_COLORS.get(element, ELEMENT_COLORS['default'])
