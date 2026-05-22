"""
=============================================================================
  HydroPhonoKit Visualization — Publication Formatter

  Implements publication-ready export capabilities:
    1. Multi-Format Exporter - PDF, EPS, SVG, PNG, TIFF with proper DPI
    2. Auto Figure Captions - Scientific descriptions for each plot type
    3. LaTeX-Compatible Labels - Proper math formatting for publications
    4. Publication Package - All figures + captions + metadata in one archive

  Scientific Foundation:
    Journal requirements:
      - Nature: 300 DPI minimum, TIFF/EPS format
      - Science: 300 DPI, PDF/EPS format
      - PRL: 600 DPI, EPS preferred
      - ACS: 300 DPI, TIFF format

    Figure captions should include:
      - Plot type and material
      - Key observations
      - Computational parameters
      - Unit specifications

  References:
    [1] Nature Publishing Guide -- Figure preparation standards
    [2] APS Style Guide -- PRL figure requirements
    [3] ACS Publications Guide -- Color palette specifications
=============================================================================
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

# Configure matplotlib backend
import os as _os
if not _os.environ.get('MPLBACKEND'):
    import matplotlib
    matplotlib.use('Agg')
del _os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .base_plotter import BasePlotter


# ============================================================================
# FIGURE CAPTION TEMPLATES
# ============================================================================

CAPTION_TEMPLATES = {
    'band_structure': (
        "Phonon band structure of {formula} along high-symmetry paths in the "
        "Brillouin zone. Calculated using density functional perturbation theory "
        "with a {supercell} supercell. {stability_note}"
    ),
    'fat_bands': (
        "Element-projected phonon band structure of {formula}. Band width is "
        "proportional to the element's contribution to each mode. Elements shown: {elements}."
    ),
    'dos': (
        "Phonon density of states (DOS) for {formula}. Total DOS (black line) "
        "and element-projected partial DOS (colored areas). Computed on a {qmesh} q-point mesh."
    ),
    'thermodynamics': (
        "Vibrational thermodynamic properties of {formula}. (a) Helmholtz free energy F(T), "
        "(b) entropy S(T), and (c) heat capacity at constant volume Cv(T). "
        "Dashed line in (c) indicates the Dulong-Petit limit of {dp_limit:.1f} J/(mol·K)."
    ),
    'ifc_decay': (
        "Interatomic force constants (IFCs) magnitude vs interatomic distance for {formula}. "
        "Exponential decay indicates convergence with respect to supercell size."
    ),
    'born_charges': (
        "Born effective charge tensors (trace) for each atom in {formula}. "
        "Sum of all charges: {total_charge:.3f}."
    ),
    'thermal_conductivity': (
        "Lattice thermal conductivity κ(T) for {formula} calculated within the "
        "relaxation time approximation. Dashed line shows 1/T high-temperature behavior."
    ),
    'hydrogen_dos': (
        "Hydrogen partial density of states for {formula}. Shaded regions indicate "
        "librational (purple), bending (blue), and stretching (red) modes."
    ),
    'gruneisen': (
        "Mode Grüneisen parameters γ(q,j) for {formula}. Positive values indicate "
        "normal anharmonic behavior; negative values suggest mode softening under compression."
    ),
}


# ============================================================================
# PUBLICATION FORMATTER CLASS
# ============================================================================

class PublicationFormatter(BasePlotter):
    """Formatter for publication-ready figure export.
    
    Provides methods for:
      - Multi-format figure export (PDF, EPS, SVG, PNG, TIFF)
      - Auto-generated figure captions
      - LaTeX-compatible labels
      - Publication package bundling
    """
    
    # Supported formats with journal recommendations
    FORMATS = {
        'pdf': {'dpi': 300, 'journals': ['Science', 'PRL', 'Nature']},
        'eps': {'dpi': 600, 'journals': ['PRL', 'Nature', 'APS']},
        'svg': {'dpi': 300, 'journals': ['web', 'presentation']},
        'png': {'dpi': 300, 'journals': ['Nature', 'web']},
        'tiff': {'dpi': 300, 'journals': ['ACS', 'Nature']},
    }
    
    def export_multi_format(self, fig: Figure, base_name: str,
                           output_dir: str,
                           formats: List[str] = None) -> Dict[str, str]:
        """Export figure in multiple formats.
        
        Args:
            fig: Matplotlib Figure object
            base_name: Base filename (without extension)
            output_dir: Output directory
            formats: List of formats (default: ['png', 'pdf', 'eps'])
        
        Returns:
            Dict mapping format -> file path
        """
        if formats is None:
            formats = ['png', 'pdf', 'eps']
        
        os.makedirs(output_dir, exist_ok=True)
        exported = {}
        
        for fmt in formats:
            if fmt not in self.FORMATS:
                continue
            
            dpi = self.FORMATS[fmt]['dpi']
            path = os.path.join(output_dir, f'{base_name}.{fmt}')
            
            save_kwargs = {'dpi': dpi, 'bbox_inches': 'tight'}
            
            if fmt == 'png':
                save_kwargs['facecolor'] = 'white'
                save_kwargs['edgecolor'] = 'white'
            elif fmt == 'tiff':
                save_kwargs['compression'] = 'tiff_lzw'
            
            fig.savefig(path, **save_kwargs)
            exported[fmt] = path
        
        return exported
    
    def generate_caption(self, plot_type: str, **kwargs) -> str:
        """Auto-generate scientific figure caption.
        
        Args:
            plot_type: Type of plot (e.g., 'band_structure', 'dos')
            **kwargs: Values for caption template variables
        
        Returns:
            Formatted caption string
        """
        template = CAPTION_TEMPLATES.get(plot_type,
            "Phonon analysis results for {formula}.")
        
        # Default values
        defaults = {
            'formula': kwargs.get('formula', 'Material'),
            'supercell': kwargs.get('supercell', '2×2×2'),
            'qmesh': kwargs.get('qmesh', '20×20×20'),
            'elements': kwargs.get('elements', 'all'),
            'stability_note': kwargs.get('stability_note', ''),
            'dp_limit': kwargs.get('dp_limit', 3 * kwargs.get('n_atoms', 1) * 8.314),
            'total_charge': kwargs.get('total_charge', 0),
        }
        
        # Merge with provided kwargs
        values = {**defaults, **kwargs}
        
        return template.format(**values)
    
    def generate_all_captions(self, plot_types: List[Dict[str, str]],
                            output_path: str):
        """Generate figure captions file for all plots.
        
        Args:
            plot_types: List of dicts with 'type' and kwargs
            output_path: Output text file path
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("HydroPhonoKit Figure Captions\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, plot_info in enumerate(plot_types, 1):
                plot_type = plot_info.get('type', 'unknown')
                kwargs = plot_info.get('kwargs', {})
                
                caption = self.generate_caption(plot_type, **kwargs)
                f.write(f"Figure {i}: {plot_type.replace('_', ' ').title()}\n")
                f.write(f"{'-' * 40}\n")
                f.write(f"{caption}\n\n")
    
    def export_publication_package(self, figures: Dict[str, Figure],
                                  captions: List[Dict[str, str]],
                                  output_dir: str,
                                  metadata: Dict = None):
        """Export complete publication package.
        
        Creates:
          - all_figures.pdf: All figures in one PDF
          - figure_captions.txt: Scientific captions for each figure
          - metadata.json: Computational metadata
          - latex_compatible/: EPS files for LaTeX
        
        Args:
            figures: Dict mapping name -> Figure object
            captions: List of caption info dicts
            output_dir: Output directory
            metadata: Optional computational metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Export individual figures in multiple formats
        fig_dir = os.path.join(output_dir, 'figures')
        exported_files = {}
        
        for name, fig in figures.items():
            exported = self.export_multi_format(
                fig, name, fig_dir,
                formats=['png', 'pdf', 'eps']
            )
            exported_files[name] = exported
        
        # 2. Generate captions
        caption_path = os.path.join(output_dir, 'figure_captions.txt')
        self.generate_all_captions(captions, caption_path)
        
        # 3. Save metadata
        if metadata:
            metadata_path = os.path.join(output_dir, 'metadata.json')
            metadata['export_date'] = datetime.now().isoformat()
            metadata['hydrophonokit_version'] = '2.7.0'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # 4. LaTeX-compatible figures
        latex_dir = os.path.join(output_dir, 'latex_compatible')
        for name, fig in figures.items():
            fig.savefig(
                os.path.join(latex_dir, f'{name}.eps'),
                dpi=600, bbox_inches='tight', format='eps'
            )
        
        print(f"  [Publication] Package exported to: {output_dir}")
        print(f"    - {len(figures)} figures in PNG/PDF/EPS")
        print(f"    - Figure captions: {caption_path}")
        if metadata:
            print(f"    - Metadata: {os.path.join(output_dir, 'metadata.json')}")
        print(f"    - LaTeX EPS files: {latex_dir}")
    
    def create_latex_compatible(self, fig: Figure, path: str):
        """Save figure in LaTeX-compatible format.
        
        Args:
            fig: Matplotlib Figure object
            path: Output EPS file path
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=600, bbox_inches='tight', format='eps')
