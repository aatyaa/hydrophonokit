"""
=============================================================================
  HydroPhonoKit Postprocessor — Core Orchestrator

  Main entry point for phonon post-processing pipeline.

  Scientific Foundation:
    The postprocessor executes a 6-phase pipeline:
      Phase 1: Data Collection (load phonopy, forces, Born charges)
      Phase 2: Force Constants (symfc/phonopy IFC computation)
      Phase 3: Band Structure & DOS (seekpath + pDOS)
      Phase 4: Thermodynamics (F, S, Cv with validations)
      Phase 5: Hydrogen Analysis (mode decomposition, if H present)
      Phase 6: Reporting (HTML summary)

    Each phase is executed independently with error isolation,
    enabling partial results even when some phases fail.

    The pipeline supports:
      - Skipping specific phases (--skip-phases)
      - Dry-run mode for validation
      - Progress callbacks for UI integration
      - Partial result saving on failure

  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy workflow
    [2] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT pipeline
=============================================================================
"""
import os
import json
import datetime
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

from .data_loader import DataLoader
from .ifc_computer import IFCComputer
from .bands_dos import BandsDOSComputer
from .thermodynamics import ThermodynamicsComputer
from .hydrogen import HydrogenAnalyzer
from .group_velocities import GroupVelocityComputer
from .debye_waller import DebyeWallerComputer
from .mode_resolved_thermo import ModeResolvedThermo
from .enhanced_export import PhononResultsExporter, EXPORT_TEMPLATES, ExportValidator
from .caching import CacheManager, CacheConfig, MemoryMonitor, get_n_jobs
from .plotting import PhononPlotter
from .reporting import ReportGenerator


# Valid phase names (updated with new scientific phases)
VALID_PHASES = {
    'data_collection',
    'force_constants',
    'bands_dos',
    'thermodynamics',
    'hydrogen',
    'group_velocities',
    'debye_waller',
    'mode_resolved_thermo',
    'plotting',
    'reporting',
    'export',
}


@dataclass
class PostprocessorConfig:
    """Configuration for the postprocessing pipeline.

    Attributes:
        skip_born_if_missing: Skip Born charge extraction if OUTCAR missing
        fallback_to_phonopy_if_symfc_fails: Use phonopy if symfc unavailable
        skip_phases: List of phase names to skip (e.g., ['hydrogen', 'plotting'])
        dry_run: Validate inputs without executing computation
        verbose: Print detailed progress messages
        save_partial_results: Save results even if some phases fail
        progress_callback: Callable(phase_name, status_dict) for progress updates
        max_retries: Number of retries for failed phases (0 = no retry)
        export_template: Export template name ('minimal', 'full', 'publication')
        use_cache: Enable caching of computed results
        cache_dir: Directory for cached files
        cache_max_age_hours: Maximum cache entry age (0 = no expiry)
        memory_limit_gb: Maximum memory usage before warning (0 = no limit)
        n_jobs: Number of parallel jobs (0 = auto-detect)
    """
    skip_born_if_missing: bool = True
    fallback_to_phonopy_if_symfc_fails: bool = True
    skip_phases: List[str] = field(default_factory=list)
    dry_run: bool = False
    verbose: bool = True
    save_partial_results: bool = True
    progress_callback: Optional[Callable[[str, Dict], None]] = None
    max_retries: int = 0
    export_template: str = 'full'
    use_cache: bool = True
    cache_dir: str = ".hydrophonokit_cache"
    cache_max_age_hours: float = 168  # 1 week
    memory_limit_gb: float = 0.0
    n_jobs: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Normalize phase names
        self.skip_phases = [p.lower().strip() for p in self.skip_phases]

        # Validate phase names
        invalid = set(self.skip_phases) - VALID_PHASES
        if invalid:
            raise ValueError(
                f"Invalid phase names in skip_phases: {invalid}\n"
                f"Valid phases: {sorted(VALID_PHASES)}"
            )

        # Validate export template
        if self.export_template not in EXPORT_TEMPLATES:
            raise ValueError(
                f"Invalid export template: {self.export_template}\n"
                f"Valid templates: {list(EXPORT_TEMPLATES.keys())}"
            )

        # dry_run implies save_partial_results
        if self.dry_run:
            self.save_partial_results = True


class PhononPostProcessor:
    """Master post-processing engine for phonon calculations.

    Executes a 6-phase pipeline with error isolation and standardized export.

    Usage:
        pp = PhononPostProcessor(workspace_dir, profile)
        results = pp.execute_pipeline()
    """

    def __init__(self, workspace_dir: str, profile, output_dir: str = None,
                 config: PostprocessorConfig = None):
        """
        Args:
            workspace_dir: Path to HydroPhonoKit workspace
            profile: MaterialProfile from analyzer
            output_dir: Override output directory (default: timestamped)
            config: Pipeline configuration
        """
        self.workspace_dir = workspace_dir
        self.profile = profile
        self.config = config or PostprocessorConfig()

        # Setup output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join(workspace_dir, f'phonon_results_{stamp}')

        if not self.config.dry_run:
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize phase components
        self.data_loader = DataLoader(workspace_dir, profile)
        self.ifc_computer = IFCComputer(profile)
        self.bands_dos = BandsDOSComputer(profile)
        self.thermo = ThermodynamicsComputer(profile)
        self.hydrogen = HydrogenAnalyzer(profile)
        self.group_velocities = GroupVelocityComputer
        self.debye_waller = DebyeWallerComputer
        self.mode_resolved_thermo = ModeResolvedThermo
        self.plotter = PhononPlotter()
        self.reporting = ReportGenerator()
        self.exporter = PhononResultsExporter(template=self.config.export_template)

        # Initialize caching and memory monitoring
        self.cache = CacheManager(CacheConfig(
            enabled=self.config.use_cache,
            cache_dir=self.config.cache_dir,
            max_age_hours=self.config.cache_max_age_hours,
        ))
        self.memory = MemoryMonitor(limit_gb=self.config.memory_limit_gb)

        # Phase status tracking
        self.phase_status: Dict[str, Dict] = {}

    def execute_pipeline(self) -> Dict:
        """Execute the full post-processing pipeline.

        Returns:
            dict: All results from each phase
        """
        print("\n" + "=" * 60)
        print(f"  PHONON POST-PROCESSING: {self.profile.formula}")
        print("=" * 60)

        if self.config.dry_run:
            print("\n  [DRY RUN] Validating inputs without execution...")
            return self._dry_run_validation()

        results = {}
        results['profile'] = self.profile
        results['output_dir'] = self.output_dir
        results['config'] = {
            'skip_phases': self.config.skip_phases,
            'max_retries': self.config.max_retries,
        }

        # Phase 1: Data Collection
        results['data_loader'] = self._execute_phase(
            'data_collection',
            self.data_loader.load
        )

        # Critical phase check
        if results['data_loader'] is None:
            print("\n  [ERROR] Data loading failed. Cannot proceed.")
            self._finalize_pipeline(results)
            return results

        phonon = results['data_loader'].get('phonon')

        # Phase 2: Force Constants
        results['ifc'] = self._execute_phase(
            'force_constants',
            lambda: self.ifc_computer.compute(phonon)
        )

        # Phase 3: Band Structure & DOS
        results['bands_dos'] = self._execute_phase(
            'bands_dos',
            lambda: self.bands_dos.compute(phonon)
        )

        # Phase 4: Thermodynamics
        results['thermodynamics'] = self._execute_phase(
            'thermodynamics',
            lambda: self.thermo.compute(phonon)
        )

        # Phase 5: Hydrogen Analysis (conditional + skip check)
        if 'H' in self.profile.elements and 'hydrogen' not in self.config.skip_phases:
            results['hydrogen'] = self._execute_phase(
                'hydrogen',
                lambda: self.hydrogen.analyze(results.get('bands_dos', {}))
            )
        elif 'hydrogen' in self.config.skip_phases:
            self.phase_status['hydrogen'] = {
                'completed': False,
                'error': 'Skipped by configuration',
                'duration_s': 0,
            }
            print("\n  [i] Skipping hydrogen analysis (configured).")

        # Scientific Enhancements (Phase 5+)
        # Group Velocities
        results['group_velocities'] = self._execute_phase(
            'group_velocities',
            lambda: self._compute_group_velocities(phonon)
        )

        # Debye-Waller Factors
        results['debye_waller'] = self._execute_phase(
            'debye_waller',
            lambda: self._compute_debye_waller(phonon)
        )

        # Mode-Resolved Thermodynamics
        results['mode_resolved_thermo'] = self._execute_phase(
            'mode_resolved_thermo',
            lambda: self._compute_mode_resolved_thermo(phonon)
        )

        # Plotting
        results['plotting'] = self._execute_phase(
            'plotting',
            lambda: self._plot_all(results)
        )

        # Phase 6: Reporting
        results['reporting'] = self._execute_phase(
            'reporting',
            lambda: self.reporting.generate(results, self.output_dir)
        )

        # Export
        results['export'] = self._execute_phase(
            'export',
            lambda: self.exporter.export(results, self.output_dir)
        )

        # Write BORN file if applicable
        if results.get('data_loader', {}).get('born_charges') is not None:
            self._write_born_file(results['data_loader'])

        self._finalize_pipeline(results)
        return results

    def _dry_run_validation(self) -> Dict:
        """Validate inputs without executing computation.

        Checks:
          - workspace_dir exists
          - phonopy_disp.yaml exists
          - Required displacement directories exist
          - Profile is valid
        """
        results = {
            'dry_run': True,
            'workspace_dir': self.workspace_dir,
            'profile': {
                'formula': getattr(self.profile, 'formula', 'Unknown'),
                'space_group': getattr(self.profile, 'space_group', 'Unknown'),
            },
            'validations': {},
        }

        # Check workspace
        ws_exists = os.path.isdir(self.workspace_dir)
        results['validations']['workspace_exists'] = ws_exists

        if ws_exists:
            # Check phonopy_disp.yaml
            yaml_path = os.path.join(self.workspace_dir, 'phonopy_disp.yaml')
            yaml_exists = os.path.exists(yaml_path)
            results['validations']['phonopy_disp_yaml'] = yaml_exists

            # Check displacement directories
            disp_dir = os.path.join(self.workspace_dir, '02_displacements')
            disp_exists = os.path.isdir(disp_dir)
            results['validations']['displacements_dir'] = disp_exists

            if disp_exists:
                import glob
                disp_folders = glob.glob(os.path.join(disp_dir, 'disp-*'))
                results['validations']['n_displacements'] = len(disp_folders)

            # Check Born charges (if expected)
            if self.profile.rec_born:
                born_outcar = os.path.join(self.workspace_dir, '01_born', 'OUTCAR')
                born_exists = os.path.exists(born_outcar)
                results['validations']['born_outcar'] = born_exists
                if not born_exists:
                    print("  [!] WARNING: Born charges expected but 01_born/OUTCAR missing.")

        # Print validation summary
        print("\n  Validation Results:")
        for check, passed in results['validations'].items():
            status = "[OK]" if passed else "[FAIL]"
            print(f"    {status} {check}: {passed}")

        all_passed = all(results['validations'].values())
        print(f"\n  Overall: {'PASSED' if all_passed else 'FAILED'}")
        print("  Workspace is ready for post-processing." if all_passed
              else "  Fix the issues above before running post-processing.")

        return results

    def _execute_phase(self, phase_name: str, func, *args, **kwargs):
        """Execute a phase with error handling, retries, and progress tracking.

        Args:
            phase_name: Name of the phase (for status tracking)
            func: Callable to execute
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result of func, or None if failed
        """
        # Check if phase should be skipped
        if phase_name in self.config.skip_phases:
            self.phase_status[phase_name] = {
                'completed': False,
                'error': 'Skipped by configuration',
                'duration_s': 0,
            }
            if self.config.verbose:
                print(f"\n  [i] Skipping phase: {phase_name}")
            return None

        # Execute with retries
        last_error = None
        for attempt in range(1, self.config.max_retries + 2):  # +1 for initial attempt
            result = self._safe_execute(phase_name, func, attempt=attempt)
            if result is not None or self.config.max_retries == 0:
                return result
            last_error = result  # Contains error info

        # All retries failed
        if self.config.verbose:
            print(f"\n  [ERROR] Phase '{phase_name}' failed after {self.config.max_retries + 1} attempts.")
        return None

    def _safe_execute(self, phase_name: str, func, attempt: int = 1):
        """Execute a single phase attempt with error handling.

        Args:
            phase_name: Name of the phase
            func: Callable to execute
            attempt: Attempt number (1 = first attempt)

        Returns:
            Result of func, or None if failed
        """
        start = time.time()
        try:
            result = func()
            duration = round(time.time() - start, 2)

            self.phase_status[phase_name] = {
                'completed': True,
                'error': None,
                'duration_s': duration,
                'attempt': attempt,
            }

            # Progress callback
            if self.config.progress_callback:
                self.config.progress_callback(phase_name, self.phase_status[phase_name])

            return result

        except Exception as e:
            duration = round(time.time() - start, 2)
            error_msg = f"{type(e).__name__}: {e}"

            self.phase_status[phase_name] = {
                'completed': False,
                'error': error_msg,
                'duration_s': duration,
                'attempt': attempt,
            }

            # Progress callback
            if self.config.progress_callback:
                self.config.progress_callback(phase_name, self.phase_status[phase_name])

            if self.config.save_partial_results:
                self._save_partial_results()

            if self.config.verbose:
                retry_msg = ""
                if attempt <= self.config.max_retries:
                    retry_msg = f" (Retrying... attempt {attempt + 1})"
                print(f"\n  [!] Phase '{phase_name}' failed: {error_msg}{retry_msg}")

            return None

    def _finalize_pipeline(self, results: Dict):
        """Print pipeline summary and save final status."""
        # Print phase status summary
        print("\n" + "=" * 60)
        print("  Pipeline Summary")
        print("=" * 60)

        n_completed = sum(1 for s in self.phase_status.values() if s['completed'])
        n_skipped = sum(1 for s in self.phase_status.values()
                       if not s['completed'] and 'Skipped' in (s.get('error') or ''))
        n_failed = sum(1 for s in self.phase_status.values()
                      if not s['completed'] and 'Skipped' not in (s.get('error') or ''))
        total_duration = sum(s['duration_s'] for s in self.phase_status.values())

        for phase, status in self.phase_status.items():
            if status['completed']:
                icon = "[OK]"
            elif 'Skipped' in (status.get('error') or ''):
                icon = "[SKIP]"
            else:
                icon = "[FAIL]"
            print(f"  {icon} {phase:20s} ({status['duration_s']:.2f}s)")

        print(f"\n  Completed: {n_completed} | Skipped: {n_skipped} | Failed: {n_failed}")
        print(f"  Total time: {total_duration:.2f}s")

        # Save final status
        if self.config.save_partial_results and self.output_dir:
            status_path = os.path.join(self.output_dir, 'pipeline_status.json')
            status_data = {
                'formula': getattr(self.profile, 'formula', 'Unknown'),
                'timestamp': datetime.datetime.now().isoformat(),
                'phases': self.phase_status,
                'summary': {
                    'completed': n_completed,
                    'skipped': n_skipped,
                    'failed': n_failed,
                    'total_duration_s': round(total_duration, 2),
                }
            }
            with open(status_path, 'w') as f:
                json.dump(status_data, f, indent=2)
            print(f"  Status saved to: {status_path}")

        if n_failed == 0 and n_completed > 0:
            print("\n[DONE] Post-processing pipeline complete.")
            print(f"    Results saved to: {self.output_dir}")
        elif n_failed > 0:
            print(f"\n[WARNING] Pipeline completed with {n_failed} failed phase(s).")
            print(f"    Partial results saved to: {self.output_dir}")

        # Print cache and memory statistics
        if self.config.use_cache:
            cache_stats = self.cache.get_stats()
            print(f"\n  Cache Statistics:")
            print(f"    Hits: {cache_stats['hits']} | Misses: {cache_stats['misses']} | Hit Rate: {cache_stats['hit_rate_pct']}%")
            print(f"    Cache Size: {cache_stats['cache_size_mb']:.1f} MB")

        mem_stats = self.memory.check()
        if mem_stats['usage_gb'] > 0.1:
            print(f"\n  Memory Usage: {mem_stats['usage_gb']:.2f} GB")

    def _plot_all(self, results: Dict):
        """Generate all plots from results."""
        print("\n[Plotting] Generating publication-grade figures...")

        bands = results.get('bands_dos')
        thermo = results.get('thermodynamics')
        h_data = results.get('hydrogen')

        if bands:
            self.plotter.plot_band_structure(
                bands['band_dict'],
                self.profile.formula,
                self.output_dir
            )

            self.plotter.plot_partial_dos(
                bands['pdos_data'],
                self.profile.formula,
                self.output_dir
            )

        if thermo:
            phonon = results.get('data_loader', {}).get('phonon')
            n_atoms = len(phonon.unitcell.numbers) if phonon else 0
            self.plotter.plot_thermodynamics(
                thermo['thermo_data'],
                self.profile.formula,
                n_atoms,
                self.output_dir
            )

        if h_data and h_data.get('decomposition'):
            bands = results.get('bands_dos', {})
            pdos_data = bands.get('pdos_data')
            if pdos_data:
                self.plotter.plot_h_modes(
                    pdos_data,
                    {
                        'peak_thz': h_data.get('peak_stretching', {}).get('freq_thz', 0),
                        'peak_cm': h_data.get('peak_stretching', {}).get('freq_cm', 0),
                    },
                    self.profile.formula,
                    self.output_dir
                )

    def _compute_group_velocities(self, phonon):
        """Compute group velocities as a scientific enhancement."""
        gv_computer = GroupVelocityComputer(phonon)
        return gv_computer.compute()

    def _compute_debye_waller(self, phonon):
        """Compute Debye-Waller factors."""
        dw_computer = DebyeWallerComputer(phonon, self.profile)
        return dw_computer.compute(temperature=300)

    def _compute_mode_resolved_thermo(self, phonon):
        """Compute mode-resolved thermodynamics."""
        mrt = ModeResolvedThermo(phonon, self.profile)
        return mrt.compute()

    def _write_born_file(self, data_result: Dict):
        """Write BORN file to results directory."""
        from ..physics import BORN_FACTOR

        born_charges = data_result.get('born_charges')
        dielectric = data_result.get('dielectric')

        if born_charges is None or dielectric is None:
            return

        born_path = os.path.join(self.output_dir, 'BORN')
        with open(born_path, 'w') as f:
            f.write(f"{BORN_FACTOR:.5f}\n")
            d = dielectric
            f.write(f"{d[0,0]:.6f} {d[0,1]:.6f} {d[0,2]:.6f} "
                    f"{d[1,0]:.6f} {d[1,1]:.6f} {d[1,2]:.6f} "
                    f"{d[2,0]:.6f} {d[2,1]:.6f} {d[2,2]:.6f}\n")
            for z in born_charges:
                f.write(f"{z[0,0]:.6f} {z[0,1]:.6f} {z[0,2]:.6f} "
                        f"{z[1,0]:.6f} {z[1,1]:.6f} {z[1,2]:.6f} "
                        f"{z[2,0]:.6f} {z[2,1]:.6f} {z[2,2]:.6f}\n")
        print(f"  --> Saved BORN to: {born_path}")

    def _save_partial_results(self):
        """Save partial results on pipeline failure."""
        if not self.config.save_partial_results:
            return
        partial_path = os.path.join(self.output_dir, 'PARTIAL_RESULTS.txt')
        with open(partial_path, 'w', encoding='utf-8') as f:
            f.write("HydroPhonoKit Partial Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Workspace: {self.workspace_dir}\n")
            f.write(f"Formula: {self.profile.formula}\n")
            f.write("\nPhase Status:\n")
            for phase, status in self.phase_status.items():
                if status['completed']:
                    completed = "[OK]"
                else:
                    completed = "[FAIL]"
                error = f" -- {status['error']}" if status.get('error') else ""
                f.write(f"  {completed} {phase}{error}\n")
