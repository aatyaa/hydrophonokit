"""
HydroPhonoKit CLI — Single entry point for the full phonon workflow.

COMMANDS:
  hydrophonokit                          → Interactive wizard (no args)
  hydrophonokit analyze  --source DIR    → Analyze material only
  hydrophonokit build    --source DIR --output DIR   → Build VASP workspace
  hydrophonokit postprocess --source DIR --workspace DIR [--output DIR]
  hydrophonokit help                     → Print this help

EXAMPLES:
  python -m hydrophonokit analyze --source C:/vasp/BaTiO3_relax
  python -m hydrophonokit build   --source C:/vasp/BaTiO3_relax --output C:/phonon/BaTiO3_ws
  python -m hydrophonokit postprocess --source C:/vasp/BaTiO3_relax --workspace C:/phonon/BaTiO3_ws
"""
import argparse
import sys
import os

# Force UTF-8 encoding for console output (fixes Windows .exe crashes with special chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'
from .analyzer import MaterialAnalyzer
from .verifier import SystemVerifier
from .hpc_profiles import get_profile, load_custom_profile, list_profiles
from .status import print_status_report
from .postprocessor.core import PhononPostProcessor
from phonopy.interface.vasp import read_vasp

# Read version from __init__.py without circular import
import importlib.metadata
try:
    VERSION = importlib.metadata.version("hydro-phonokit")
except Exception:
    VERSION = "2.7.0"

BANNER = f"""
 +========================================================+
 |                                                        |
 |   ____  __                          __ __ _ __        |
 |  / __ \\/ /_  ____  ____  ____     / //_/(_) /_       |
 | / /_/ / __ \\/ __ \\/ __ \\/ __ \\   / ,<  / / __/      |
 |/ ____/ / / / /_/ / / / / /_/ /  / /| |/ / /_        |
 |/_/   /_/ /_/\\____/_/ /_/\\____/  /_/ |_/_/\\__/       |
 |                                        v{VERSION}             |
 |         Material-Aware Phonon Workflow Engine         |
 |                                                        |
 +========================================================+"""

USAGE = f"""
  +==================================================================+
  |        HYDROPHONOKIT v{VERSION} - COMPLETE COMMAND REFERENCE                |
  +==================================================================+
  |                                                                   |
  |  CORE WORKFLOW (VASP -> Phonons -> Analysis -> Visualization):    |
  |    analyze      Analyze relaxed VASP directory (read-only report)|
  |    build        Generate phonon workspace (displacements+INCARs) |
  |    status       Monitor workspace convergence (Born/Displacements)|
  |    postprocess  Full analysis: bands, DOS, thermo, H-analysis    |
  |    visualize    Generate publication-ready figures from results  |
  |                                                                   |
  |  ADVANCED PHYSICS:                                               |
  |    elastic      Elastic constants (C11, C12, C44), sound vel.   |
  |    qha          Thermal expansion, C_p, Gruneisen parameter      |
  |    anharmonic   Phonon linewidths, lifetimes, thermal cond.      |
  |    hstorage     H-storage: dH, dS, T_des, DOE target evaluation  |
  |                                                                   |
  |  QUICK USAGE:                                                    |
  |    python -m hydrophonokit                                           |
  |       -> Interactive wizard (guides you step-by-step)            |
  |                                                                   |
  |    python -m hydrophonokit analyze  --source <VASP_DIR>              |
  |    python -m hydrophonokit build    --source <DIR> -o <DIR> --hpc X  |
  |    python -m hydrophonokit status   --workspace <WORKSPACE_DIR>      |
  |    python -m hydrophonokit postprocess -s <DIR> -w <WORKSPACE>       |
  |    python -m hydrophonokit visualize   -s <DIR> -w <WORKSPACE>       |
  |    python -m hydrophonokit elastic  --source <VASP_DIR>              |
  |    python -m hydrophonokit qha      --source <QHA_JSON>              |
  |    python -m hydrophonokit anharmonic --source <VASP_DIR>            |
  |    python -m hydrophonokit hstorage --hydride <DIR> --dehydride <D>  |
  |                                                                   |
  |  HPC PROFILE OPTIONS (for build):                                |
  |    --hpc PROFILE    Target HPC: bridges2|roar|generic|custom|list|
  |    --hpc-config F   YAML config for custom clusters              |
  |    --email EMAIL    Email for SLURM job notifications            |
  |                                                                   |
  |  POSTPROCESS OPTIONS:                                            |
  |    --skip-phases LIST     Comma-separated phases to skip        |
  |    --dry-run              Validate workspace without computing   |
  |    --max-retries N        Retry failed phases N times (def: 0)   |
  |    --export-template T    Export: minimal|full|publication      |
  |    --no-cache             Disable result caching                 |
  |    --cache-dir DIR        Custom cache directory                 |
  |    --memory-limit GB      Memory usage limit (0 = unlimited)     |
  |    --n-jobs N             Parallel jobs (0 = auto-detect)        |
  |    -q, --quiet            Suppress verbose output                |
  |                                                                   |
  |  VISUALIZATION OPTIONS:                                          |
  |    --figures DIR          Output directory for figures           |
  |    --theme THEME          nature|science|prl|acs|pres|minimal    |
  |    --format FORMAT        Output: png,pdf,eps,svg,tiff           |
  |    --no-interactive       Skip HTML interactive plots            |
  |    --all-figures          Generate ALL possible figure types     |
  |                                                                   |
  |  COMMON OPTIONS:                                                 |
  |    --source, -s       Path to relaxed VASP directory             |
  |    --output, -o       Output directory                           |
  |    --workspace, -w    Path to completed phonon workspace         |
  |    --help, -h         Show this help message                     |
  |                                                                   |
  |  FULL WORKFLOW EXAMPLE:                                          |
  |    1. hydrophonokit analyze --source ./relax_dir                     |
  |    2. hydrophonokit build   --source ./relax_dir -o ./ws --hpc bridges2|
  |    3. Upload ./ws to HPC, run: bash submit_born.sh               |
  |    4. After Born: bash submit_all_disps.sh                       |
  |    5. hydrophonokit status  --workspace ./ws                         |
  |    6. hydrophonokit postprocess -s ./relax_dir -w ./ws               |
  |    7. hydrophonokit visualize   -s ./relax_dir -w ./ws               |
  |    8. Results: phonon_figures/ (PNG, PDF, EPS, HTML interactive) |
  |                                                                   |
  +==================================================================+"""


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class HydroPhonoKitError(Exception):
    """Base exception for HydroPhonoKit CLI errors."""
    pass

class SourceDirectoryError(HydroPhonoKitError):
    """Raised when source directory validation fails."""
    pass

class WorkspaceDirectoryError(HydroPhonoKitError):
    """Raised when workspace directory validation fails."""
    pass


# ============================================================================
# HELPERS
# ============================================================================

def _require_source(args):
    """Validate and return source directory path.

    Raises:
        SourceDirectoryError: If source is missing or invalid
    """
    if not args.source:
        raise SourceDirectoryError(
            f"--source is required for '{args.action}'.\n"
            f"    Usage: python -m hydrophonokit {args.action} --source <PATH>"
        )
    src = os.path.abspath(args.source)
    missing = [f for f in ['CONTCAR', 'INCAR', 'OUTCAR', 'POTCAR']
               if not os.path.exists(os.path.join(src, f))]
    if missing:
        raise SourceDirectoryError(
            f"Source directory is missing: {', '.join(missing)}\n"
            f"    Directory: {src}"
        )
    return src


def _require_workspace(args):
    """Validate and return workspace directory path.

    Raises:
        WorkspaceDirectoryError: If workspace is missing or invalid
    """
    if not args.workspace:
        raise WorkspaceDirectoryError(
            "--workspace is required for 'postprocess'.\n"
            "    Usage: python -m hydrophonokit postprocess --source <DIR> --workspace <DIR>"
        )
    ws = os.path.abspath(args.workspace)
    if not os.path.isdir(ws):
        raise WorkspaceDirectoryError(
            f"Workspace directory does not exist: {ws}"
        )
    return ws


def _analyze_source(source):
    """Run full 9-layer analysis on source and return (analyzer, profile)."""
    analyzer = MaterialAnalyzer(source)
    profile  = analyzer.analyze()
    return analyzer, profile


# ============================================================================
# ACTIONS
# ============================================================================

def cmd_analyze(args):
    """analyze: deep material scan, prints profile + phonon plan."""
    source = _require_source(args)
    analyzer, profile = _analyze_source(source)
    analyzer.print_profile()
    analyzer.print_recommendations()


def cmd_build(args):
    """build: analyze + generate VASP workspace ready for HPC."""
    source = _require_source(args)

    if not args.output:
        print("  [!] --output is required for 'build'.")
        print("      Usage: python -m hydrophonokit build --source <DIR> --output <DIR>")
        sys.exit(1)

    output = os.path.abspath(args.output)

    # Check overwrite
    if os.path.exists(output) and os.listdir(output):
        ans = input(f"\n  [?] '{output}' exists and is not empty. Overwrite? (y/n): ").strip().lower()
        if ans != 'y':
            print("  Aborted.")
            sys.exit(0)

    # Resolve HPC profile
    hpc_profile = _resolve_hpc_profile(args)

    analyzer, profile = _analyze_source(source)
    analyzer.print_profile()
    analyzer.print_recommendations()

    print()
    _run_build_pipeline(source, output, profile, hpc_profile=hpc_profile)


def cmd_status(args):
    """status: Monitor convergence status of generated workspace."""
    workspace = _require_workspace(args)
    print_status_report(workspace)


def _resolve_hpc_profile(args):
    """Resolve HPC profile from CLI arguments.

    Returns:
        dict or None: HPC profile, or None if not specified
    """
    hpc_name = getattr(args, 'hpc', None)
    hpc_config = getattr(args, 'hpc_config', None)

    if not hpc_name:
        return None

    if hpc_name == 'list':
        print(list_profiles())
        sys.exit(0)

    if hpc_name == 'custom':
        if not hpc_config:
            print("  [!] --hpc-config is required when --hpc custom is used.")
            sys.exit(1)
        profile = load_custom_profile(hpc_config)
        print(f"  [HPC] Loaded custom profile: {profile['name']}")
        return profile

    # Apply email override if provided
    profile = get_profile(hpc_name)
    email = getattr(args, 'email', None)
    if email:
        profile['email'] = email
    print(f"  [HPC] Using profile: {profile['name']} "
          f"({profile['cores_per_node']} cores, {profile['partition']})")
    return profile


def cmd_postprocess(args):
    """postprocess: parse completed VASP phonon results and compute all properties."""
    source    = _require_source(args)
    workspace = _require_workspace(args)

    # Optional custom output dir inside workspace
    output_dir = os.path.abspath(args.output) if args.output else None

    print("\n  Building MaterialProfile from source directory...")
    _, profile = _analyze_source(source)

    # Build config from CLI args
    config = _build_postprocessor_config(args)

    pp = PhononPostProcessor(workspace, profile, output_dir=output_dir, config=config)
    pp.execute_pipeline()


def _build_postprocessor_config(args) -> 'PostprocessorConfig':
    """Build PostprocessorConfig from CLI arguments.

    Args:
        args: Parsed argparse namespace

    Returns:
        PostprocessorConfig instance
    """
    from .postprocessor.core import PostprocessorConfig

    skip_phases = []
    if args.skip_phases:
        skip_phases = [p.strip() for p in args.skip_phases.split(',')]

    return PostprocessorConfig(
        skip_phases=skip_phases,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        verbose=not args.quiet,
        save_partial_results=True,
        export_template=args.export_template,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir or ".hydrophonokit_cache",
        memory_limit_gb=args.memory_limit,
        n_jobs=args.n_jobs,
    )


# ============================================================================
# BUILD PIPELINE (called by both interactive and cmd_build)
# ============================================================================

def _run_build_pipeline(source, output, profile, hpc_profile=None):
    print("=" * 56)
    print("  Generating Phonon Workspace ...")
    if hpc_profile:
        print(f"  HPC Target: {hpc_profile['name']}")
    print("=" * 56)

    unitcell  = read_vasp(os.path.join(source, 'CONTCAR'))
    generator = PhononGenerator(
        unitcell=unitcell,
        output_dir=output,
        profile=profile,
        source_dir=source,
        hpc_profile=hpc_profile
    )
    n_disp = generator.generate()

    print("\n" + "=" * 56)
    print("  Verifying workspace integrity ...")
    print("=" * 56)
    verifier = SystemVerifier(output, n_disp)
    verifier.verify_integrity()

    born_str = "+ 1 Born charge calculation" if profile.rec_born else "(no Born charges)"
    hpc_str = f" [{hpc_profile['name']}]" if hpc_profile else ""

    print("\n" + "=" * 56)
    print(f"  ✓ Workspace Generation Complete{hpc_str}")
    print("=" * 56)
    print(f"  Location : {output}")
    print(f"  Jobs     : {n_disp} displacements {born_str}")
    if hpc_profile:
        print(f"  Platform : {hpc_profile['name']} ({hpc_profile['cores_per_node']} cores/node)")
        print(f"  Walltime : Born={hpc_profile['born_walltime']}, Disp={hpc_profile['disp_walltime']}")
    print("=" * 56)
    print(f"""
  ► Next steps:
    1. Upload '{os.path.basename(output)}/' to supercomputer
    {"2. Run:  bash submit_born.sh" if profile.rec_born else ""}
    {"3" if profile.rec_born else "2"}. Run:  bash submit_all_disps.sh
    {"4" if profile.rec_born else "3"}. Monitor progress:
       hydrophonokit status --workspace {output}
    {"5" if profile.rec_born else "4"}. After completion:
       hydrophonokit postprocess --source {source} --workspace {output}

  See PHONON_PLAN.txt inside the workspace for full details.
""")


# ============================================================================
# INTERACTIVE WIZARD  (triggered when no args given)
# ============================================================================

def interactive_mode():
    """Interactive wizard for HydroPhonoKit CLI."""
    
    # Main loop to keep returning to the menu
    while True:
        print("\n  ╔══════════════════════════════════════════════════════╗")
        print("  ║     Welcome to HydroPhonoKit v" + VERSION + " Interactive Wizard!    ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print("  Run 'hydrophonokit --help' for direct command usage.\n")

        print("  ═══ CORE WORKFLOW ═══")
        print("    1. 📊 Analyze a relaxed VASP directory (read-only report)")
        print("    2. 🏗️  Build a phonon workspace (displacements + INCARs)")
        print("    3. 🔎 Monitor status of running VASP calculations")
        print("    4. 📈 Post-process completed VASP phonon results")
        print("    5. 🎨 Generate publication-ready figures (visualize)")
        print()
        print("  ═══ ADVANCED PHYSICS ═══")
        print("    6. 🔬 Extract elastic constants (C11, C12, C44, sound vel.)")
        print("    7. 🌡️  Quasi-Harmonic Approximation (thermal expansion)")
        print("    8. 📉 Anharmonic properties (linewidths, lifetimes, κ)")
        print("    9. 💧 Hydrogen storage (ΔH, ΔS, T_des, DOE targets)")
        print()
        print("  ═══ UTILITIES ═══")
        print("    10. ⚙️  Postprocess with custom options (skip/cache/export)")
        print("    11. ❌ Exit\n")

        while True:
            choice = input("  Select (1-11): ").strip()
            if choice in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'):
                break
            print("  [!] Enter a number from 1 to 11.")

        if choice == '11':
            print("  Goodbye!")
            return

        # Helper to run command and wait for user to return to menu
        def run_and_wait(func_name, func):
            try:
                print("\n" + "="*56)
                print(f"  Running: {func_name}")
                print("="*56 + "\n")
                func()
            except HydroPhonoKitError as e:
                print(f"\n  [!] Error: {e}")
            except Exception as e:
                print(f"\n  [!] Unexpected error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print("\n" + "-"*56)
                input("  Press Enter to return to the main menu...")

        # Option 1: Analyze
        if choice == '1':
            def do_analyze():
                print("\n" + "-" * 56)
                print("  Source VASP Directory")
                print("  Required files: CONTCAR, INCAR, OUTCAR, POTCAR")
                print("-" * 56)
                source = _get_interactive_source()
                analyzer, profile = _analyze_source(source)
                analyzer.print_profile()
                analyzer.print_recommendations()
            run_and_wait("Analyze Material", do_analyze)

        # Option 2: Build
        elif choice == '2':
            def do_build():
                print("\n" + "-" * 56)
                print("  Source VASP Directory")
                print("-" * 56)
                source = _get_interactive_source()
                analyzer, profile = _analyze_source(source)
                analyzer.print_profile()
                
                print("\n  Accept plan?")
                mod = input("  (y to accept, m to modify, n to cancel): ").strip().lower()
                if mod == 'n': return
                if mod == 'm':
                    _modify_plan_interactive(profile)
                
                # HPC profile selection
                print("\n" + "-" * 56)
                print("  HPC Platform Selection")
                print("-" * 56)
                print("    1. Bridges-2 (PSC, 128 cores/node)")
                print("    2. ROAR Collab (ICDS, 32 cores/node)")
                print("    3. Generic SLURM cluster")
                print("    4. Skip (use default templates)")
                hpc_choice = input("\n  Select HPC (1-4): ").strip()
                hpc_profile = None
                if hpc_choice == '1':
                    hpc_profile = get_profile('bridges2')
                    email = input("  Email for SLURM notifications (Enter to skip): ").strip()
                    if email:
                        hpc_profile['email'] = email
                elif hpc_choice == '2':
                    hpc_profile = get_profile('roar')
                    email = input("  Email for SLURM notifications (Enter to skip): ").strip()
                    if email:
                        hpc_profile['email'] = email
                elif hpc_choice == '3':
                    hpc_profile = get_profile('generic')
                if hpc_profile:
                    print(f"  [HPC] → {hpc_profile['name']} selected")

                print("\n" + "-" * 56)
                print("  Output Workspace Directory")
                print("-" * 56)
                output = _get_interactive_output()
                _run_build_pipeline(source, output, profile, hpc_profile=hpc_profile)
            run_and_wait("Build Workspace", do_build)

        # Option 3: Status
        elif choice == '3':
            def do_status():
                print("\n" + "-" * 56)
                print("  Target Workspace Directory")
                print("-" * 56)
                workspace = _get_interactive_workspace()
                print_status_report(workspace)
            run_and_wait("Monitor Status", do_status)

        # Option 4: Postprocess
        elif choice == '4':
            def do_postprocess():
                print("\n" + "-" * 56)
                print("  Source VASP Directory")
                print("-" * 56)
                source = _get_interactive_source()
                
                print("\n" + "-" * 56)
                print("  Completed Phonon Workspace Directory")
                print("-" * 56)
                ws = _get_interactive_workspace()
                
                pp = PhononPostProcessor(ws, _analyze_source(source)[1])
                pp.execute_pipeline()
            run_and_wait("Postprocess Results", do_postprocess)

        # Option 5: Visualize
        elif choice == '5':
            def do_visualize():
                print("\n" + "-" * 56)
                print("  Source VASP Directory")
                print("-" * 56)
                source = _get_interactive_source()
                
                print("\n" + "-" * 56)
                print("  Completed Phonon Workspace Directory")
                print("-" * 56)
                ws = _get_interactive_workspace()
                
                # Ask for theme
                print("\n  Select theme:")
                print("    1. Nature (default)")
                print("    2. Science")
                print("    3. PRL")
                print("    4. ACS")
                print("    5. Presentation (dark)")
                print("    6. Minimal")
                theme_choice = input("\n  Select (1-6, default 1): ").strip()
                theme_map = {'1': 'nature', '2': 'science', '3': 'prl', '4': 'acs', '5': 'presentation', '6': 'minimal'}
                theme = theme_map.get(theme_choice, 'nature')
                
                # Mock args for cmd_visualize
                class Args:
                    pass
                args = Args()
                args.source = source
                args.workspace = ws
                args.figures = None
                args.theme = theme
                args.format = None
                args.no_interactive = False
                args.all_figures = False
                args.export_template = 'full'
                
                cmd_visualize(args)
            run_and_wait("Visualize Results", do_visualize)

        # Option 6: Elastic
        elif choice == '6':
            def do_elastic():
                print("\n" + "-" * 56)
                print("  Source VASP Directory")
                print("-" * 56)
                source = _get_interactive_source()
                class Args:
                    pass
                args = Args()
                args.source = source
                args.workspace = None
                cmd_elastic(args)
            run_and_wait("Extract Elastic Constants", do_elastic)

        # Option 7: QHA
        elif choice == '7':
            print("\n" + "-" * 56)
            print("  Quasi-Harmonic Approximation")
            print("-" * 56)
            print("  QHA requires a JSON input file with E(V) data.")
            print("  Usage: hydrophonokit qha --source qha_input.json")
            print("\n  See documentation for JSON format details.")
            input("\n  Press Enter to return to the main menu...")

        # Option 8: Anharmonic
        elif choice == '8':
            def do_anharmonic():
                print("\n" + "-" * 56)
                print("  Source VASP Directory")
                print("-" * 56)
                source = _get_interactive_source()
                class Args:
                    pass
                args = Args()
                args.source = source
                cmd_anharmonic(args)
            run_and_wait("Anharmonic Properties", do_anharmonic)

        # Option 9: H-Storage
        elif choice == '9':
            print("\n" + "-" * 56)
            print("  Hydrogen Storage Thermodynamics")
            print("-" * 56)
            print("  H-storage requires hydride and dehydride directories.")
            print("  Usage: hydrophonokit hstorage --hydride DIR --dehydride DIR")
            input("\n  Press Enter to return to the main menu...")

        # Option 10: Custom Postprocess
        elif choice == '10':
            print("\n" + "-" * 56)
            print("  Custom Postprocess Options")
            print("-" * 56)
            print("  For advanced postprocessing with custom options, use:")
            print()
            print("  hydrophonokit postprocess \\")
            print("      --source <VASP_DIR> \\")
            print("      --workspace <WORKSPACE_DIR> \\")
            print("      --skip-phases hydrogen,plotting \\")
            print("      --export-template publication \\")
            print("      --no-cache \\")
            print("      --n-jobs 4")
            print()
            print("  Available options:")
            print("    --skip-phases     Skip: data_collection,force_constants,")
            print("                      bands_dos,thermodynamics,hydrogen,")
            print("                      group_velocities,debye_waller,")
            print("                      mode_resolved_thermo,plotting,")
            print("                      reporting,export")
            print("    --dry-run         Validate without computing")
            print("    --max-retries N   Retry failed phases")
            print("    --export-template minimal|full|publication")
            print("    --no-cache        Disable caching")
            print("    --cache-dir DIR   Custom cache directory")
            print("    --memory-limit GB Memory limit (0 = unlimited)")
            print("    --n-jobs N        Parallel jobs (0 = auto)")
            print("    -q, --quiet       Suppress verbose output")
            input("\n  Press Enter to return to the main menu...")

def _get_interactive_source():
    """Helper to get source path interactively."""
    while True:
        source = input("\n  Enter path: ").strip().strip('"').strip("'")
        if not source:
            print("  [!] Path cannot be empty.")
            continue
        source = os.path.abspath(source)
        if not os.path.isdir(source):
            print(f"  [!] Directory not found: {source}")
            continue
        missing = [f for f in ['CONTCAR', 'INCAR', 'OUTCAR', 'POTCAR']
                   if not os.path.exists(os.path.join(source, f))]
        if missing:
            print(f"  [!] Missing files: {', '.join(missing)}")
            continue
        print(f"  [✓] {source}")
        return source

def _get_interactive_workspace():
    """Helper to get workspace path interactively."""
    while True:
        ws = input("\n  Enter workspace path: ").strip().strip('"').strip("'")
        if not ws:
            print("  [!] Path cannot be empty.")
            continue
        ws = os.path.abspath(ws)
        if not os.path.isdir(ws):
            print(f"  [!] Directory not found: {ws}")
            continue
        print(f"  [✓] {ws}")
        return ws

def _get_interactive_output():
    """Helper to get output path interactively."""
    while True:
        output = input("\n  Enter output path: ").strip().strip('"').strip("'")
        if not output:
            print("  [!] Path cannot be empty.")
            continue
        output = os.path.abspath(output)
        if os.path.exists(output) and os.listdir(output):
            ans = input(f"  [!] '{output}' exists. Overwrite? (y/n): ").strip().lower()
            if ans != 'y':
                continue
        print(f"  [✓] {output}")
        return output


def _modify_plan_interactive(profile):
    p = profile
    print("\n  ── Modifiable Parameters ──────────────────────────")
    print(f"  1. Supercell         [{p.rec_supercell[0]}×{p.rec_supercell[1]}×{p.rec_supercell[2]}]")
    print(f"  2. Displacement      [{p.rec_displacement} Å]")
    print(f"  3. Born charges      [{'Yes' if p.rec_born else 'No'}]")
    print(f"  4. vdW correction    [{('IVDW=' + str(p.rec_ivdw)) if p.rec_vdw else 'Off'}]")
    print(f"  5. Done\n")

    while True:
        sel = input("  Select (1-5): ").strip()
        if sel == '1':
            d = input("  New supercell (e.g. 2 2 2): ").strip().split()
            if len(d) == 3 and all(x.isdigit() for x in d):
                p.rec_supercell = [int(x) for x in d]
                p.sc_atoms = p.n_atoms * (p.rec_supercell[0] * p.rec_supercell[1] * p.rec_supercell[2])
                print(f"  [✓] Supercell → {p.rec_supercell[0]}×{p.rec_supercell[1]}×{p.rec_supercell[2]} ({p.sc_atoms} atoms)")
        elif sel == '2':
            d = input("  New displacement (Å): ").strip()
            try:
                v = float(d)
                if 0.001 <= v <= 0.1:
                    p.rec_displacement = v
                    print(f"  [✓] Displacement → {v} Å")
                else:
                    print("  [!] Valid range: 0.001 – 0.100 Å")
            except ValueError:
                print("  [!] Enter a number.")
        elif sel == '3':
            d = input("  Enable Born charges? (y/n): ").strip().lower()
            p.rec_born = (d == 'y')
            print(f"  [✓] Born → {'Yes' if p.rec_born else 'No'}")
        elif sel == '4':
            d = input("  IVDW value (0=off, 11=D3, 13=D4): ").strip()
            try:
                v = int(d)
                p.rec_ivdw = v
                p.rec_vdw  = v > 0
                print(f"  [✓] vdW → {'IVDW=' + str(v) if v > 0 else 'Off'}")
            except ValueError:
                print("  [!] Enter an integer.")
        elif sel == '5':
            break


# ============================================================================
# ELASTIC CONSTANTS
# ============================================================================

def cmd_elastic(args):
    """elastic: Extract elastic constants from phonon dispersion."""
    source = _require_source(args)

    print("\n" + "-" * 56)
    print("  Elastic Constants: Phonon-Based Extraction")
    print("-" * 56 + "\n")

    # Load material profile for density
    _, profile = _analyze_source(source)

    # Find workspace
    workspace = args.workspace if args.workspace else source
    if not os.path.isdir(workspace):
        raise WorkspaceDirectoryError(f"No phonon workspace found at: {workspace}")

    print(f"  Using workspace: {workspace}")

    # Load phonon data
    try:
        import phonopy
        yaml_path = os.path.join(workspace, 'phonopy_disp.yaml')
        if os.path.exists(yaml_path):
            phonon = phonopy.load(yaml_path)
            print(f"  [✓] Loaded phonon data from phonopy_disp.yaml")
        else:
            raise FileNotFoundError("No phonopy_disp.yaml found. Run 'hydrophonokit build' first.")

        # Extract elastic constants
        from .elastic import ElasticConstantsExtractor

        extractor = ElasticConstantsExtractor(phonon, profile)
        result = extractor.extract()

        print(result.summary())

        # Save results
        import json
        output_path = os.path.join(workspace, 'elastic_constants.json')
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n  [✓] Results saved to: {output_path}")

    except FileNotFoundError as e:
        raise SourceDirectoryError(str(e)) from e
    except Exception as e:
        print(f"\n  [!] Elastic extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# QUASI-HARMONIC APPROXIMATION
# ============================================================================

def cmd_qha(args):
    """qha: Run Quasi-Harmonic Approximation analysis."""
    print("\n" + "-" * 56)
    print("  Quasi-Harmonic Approximation: Thermal Expansion")
    print("-" * 56 + "\n")

    # QHA requires input data (volumes, energies, free energies)
    # This is typically provided as a JSON file or directory structure
    if not args.source:
        raise SourceDirectoryError(
            "--source is required for QHA. Provide path to QHA input data.\n"
            "    Format: JSON file with volumes, energies_DFT, free_energies_T"
        )

    import json
    source = os.path.abspath(args.source)

    # Load input data
    if source.endswith('.json'):
        with open(source, 'r') as f:
            data = json.load(f)
        print(f"  [QHA] Loaded input from: {source}")
    else:
        raise SourceDirectoryError("QHA input must be a JSON file")

    # Validate input
    required_keys = ['volumes', 'energies_DFT', 'free_energies_T']
    for key in required_keys:
        if key not in data:
            raise SourceDirectoryError(f"Missing required key in input: {key}")

    formula = data.get('formula', 'Unknown')
    volumes = data['volumes']
    energies_DFT = data['energies_DFT']
    free_energies_T = data['free_energies_T']

    print(f"  [QHA] Material: {formula}")
    print(f"  [QHA] Volumes: {len(volumes)}")
    print(f"  [QHA] T-range: 0-1000 K")

    # Build volume data
    volume_data = []
    for i, (V, E) in enumerate(zip(volumes, energies_DFT)):
        F_T = free_energies_T[i] if i < len(free_energies_T) else free_energies_T[-1]
        volume_data.append({
            'volume': V,
            'energy_DFT': E,
            'free_energies_T': np.asarray(F_T),
        })

    # Run QHA
    from .qha import QHAEngine

    engine = QHAEngine(volume_data, formula=formula)
    result = engine.execute()

    print(result.summary())

    # Save results
    output_dir = args.output if args.output else os.path.dirname(source)
    engine.plot_results(output_dir)


# ============================================================================
# ANHARMONIC PROPERTIES
# ============================================================================

def cmd_anharmonic(args):
    """anharmonic: Compute phonon linewidths, lifetimes, thermal conductivity."""
    source = _require_source(args)

    print("\n" + "-" * 56)
    print("  Anharmonic Phonon Properties: Linewidths & Lifetimes")
    print("-" * 56 + "\n")

    # Load material profile
    _, profile = _analyze_source(source)

    # Find workspace
    workspace = args.workspace if args.workspace else source
    if not os.path.isdir(workspace):
        raise WorkspaceDirectoryError(f"No phonon workspace found at: {workspace}")

    print(f"  Using workspace: {workspace}")

    try:
        import phonopy
        yaml_path = os.path.join(workspace, 'phonopy_disp.yaml')
        if os.path.exists(yaml_path):
            phonon = phonopy.load(yaml_path)
            print(f"  [✓] Loaded phonon data from phonopy_disp.yaml")
        else:
            raise FileNotFoundError("No phonopy_disp.yaml found.")

        # Compute anharmonic properties
        from .anharmonic import AnharmonicCalculator

        calc = AnharmonicCalculator(phonon, profile)
        result = calc.compute()

        print(result.summary())

        # Save results and plots
        output_dir = args.output if args.output else workspace
        calc.plot_results(output_dir)

        print(f"\n  [✓] Anharmonic analysis complete.")

    except FileNotFoundError as e:
        raise SourceDirectoryError(str(e)) from e
    except Exception as e:
        print(f"\n  [!] Anharmonic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# HYDROGEN STORAGE
# ============================================================================

def cmd_hstorage(args):
    """hstorage: Hydrogen storage thermodynamics analysis."""
    print("\n" + "-" * 56)
    print("  Hydrogen Storage Thermodynamics & DOE Targets")
    print("-" * 56 + "\n")

    # Parse required arguments
    if not args.hydride or not args.dehydride:
        raise SourceDirectoryError(
            "--hydride and --dehydride are required.\n"
            "    Usage: python -m hydrophonokit hstorage --hydride DIR --dehydride DIR"
        )

    hydride_path = os.path.abspath(args.hydride)
    dehydride_path = os.path.abspath(args.dehydride)

    # Load hydride data
    print(f"  Loading hydride: {hydride_path}")
    hydride_data = _load_h_storage_data(hydride_path, "hydride")

    # Load dehydride data
    print(f"  Loading dehydride: {dehydride_path}")
    dehydride_data = _load_h_storage_data(dehydride_path, "dehydride")

    # H2 energy
    h2_energy = args.h2_energy if args.h2_energy is not None else -6.77
    hydride_data['h2_energy'] = h2_energy

    # n_h2
    n_h = hydride_data.get('n_h', 0)
    n_h2 = args.n_h2 if args.n_h2 is not None else None

    print(f"\n  [✓] Loaded: {hydride_data['formula']} → {dehydride_data['formula']} + {n_h2 or n_h/2} H2")

    # Run analysis
    from .h_storage import HStorageAnalyzer

    analyzer = HStorageAnalyzer(hydride_data, dehydride_data, n_h2=n_h2)
    result = analyzer.execute()

    print(result.summary())

    # Save results and plots
    output_dir = args.output if args.output else os.path.join(os.getcwd(), 'h_storage_results')
    analyzer.plot_results(output_dir)

    print(f"\n  [✓] H-storage analysis complete.")


def _load_h_storage_data(path, label):
    """Load H-storage data from directory or JSON file.

    Tries:
        1. JSON file (h_storage_data.json or input.json)
        2. Phonopy workspace with thermodynamic_properties.dat
    """
    import json

    # Try JSON first
    json_files = ['h_storage_data.json', 'input.json', 'data.json']
    for jf in json_files:
        jpath = os.path.join(path, jf)
        if os.path.exists(jpath):
            with open(jpath, 'r') as f:
                data = json.load(f)
            print(f"  [✓] Loaded {label} data from: {jpath}")
            return data

    # Try thermodynamic_properties.dat (from HydroPhonoKit postprocessing)
    thermo_path = os.path.join(path, 'thermodynamic_properties.dat')
    if os.path.exists(thermo_path):
        # Parse the file
        temperatures = []
        free_energies = []
        entropies = []

        with open(thermo_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    temperatures.append(float(parts[0]))
                    free_energies.append(float(parts[1]))
                    entropies.append(float(parts[2]))

        if temperatures:
            return {
                'formula': os.path.basename(path),
                'free_energy_T': free_energies,
                'entropy_T': entropies,
                'temperatures': temperatures,
            }

    raise SourceDirectoryError(
        f"No valid {label} data found in {path}.\n"
        f"    Expected: h_storage_data.json, input.json, or thermodynamic_properties.dat"
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def cmd_visualize(args):
    """visualize: Generate publication-ready figures from completed phonon results."""
    source = _require_source(args)

    # Workspace is optional but recommended for complete results
    workspace = args.workspace if args.workspace else source

    print("\n" + "-" * 56)
    print("  Phonon Visualization Engine")
    print("-" * 56 + "\n")

    print(f"  Source: {source}")
    print(f"  Workspace: {workspace}")
    print(f"  Theme: {args.theme}")

    # Load material profile
    print("\n  Loading material profile...")
    _, profile = _analyze_source(source)
    print(f"  Formula: {profile.formula}")

    # Determine output directory
    output_dir = args.figures
    if not output_dir:
        if os.path.isdir(workspace):
            output_dir = os.path.join(workspace, 'phonon_figures')
        else:
            output_dir = os.path.abspath('phonon_figures')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}")

    # Try to load existing results from workspace or postprocess output
    results = _load_existing_results(source, workspace, profile)
    
    if results is None:
        print("\n  [!] No results found. Run 'hydrophonokit postprocess' first,")
        print("      or provide a completed workspace with VASP outputs.")
        return

    print(f"\n  Data loaded: {sum(1 for v in results.values() if v is not None)} sections")

    # Create visualization factory
    from .visualization import PhononFigureFactory
    factory = PhononFigureFactory(results, theme=args.theme)

    # Determine output formats
    formats = None
    if args.format:
        formats = [f.strip() for f in args.format.split(',')]

    # Generate figures
    print("\n  Generating figures...")
    print("-" * 56)

    plots_generated = 0

    # Band structure
    if results.get('bands_dos') and results['bands_dos'].get('band_dict'):
        print("\n  [Bands] Generating band structure figures...")
        try:
            factory.plot_band_structure(save=True, output_dir=output_dir)
            print("    [OK] phonon_band_structure.png")
            plots_generated += 1

            # Zoomed views
            for fmin, fmax in [(0, 5), (5, 20), (20, 50)]:
                factory.plot_zoomed_band_structure((fmin, fmax), save=True, output_dir=output_dir)
                print("    [OK] band_zoomed_%d_%d.png" % (fmin, fmax))
                plots_generated += 1
        except Exception as e:
            print("    [FAIL] %s" % str(e)[:60])

        # Fat bands if pDOS available
        if results['bands_dos'].get('pdos_data'):
            try:
                factory.plot_fat_bands(save=True, output_dir=output_dir)
                print("    [OK] phonon_fat_bands.png")
                plots_generated += 1
            except Exception as e:
                print("    [FAIL] %s" % str(e)[:60])
    else:
        print("\n  [Bands] No band structure data available")

    # DOS
    if results.get('bands_dos') and results['bands_dos'].get('pdos_data'):
        print("\n  [DOS] Generating DOS figures...")
        try:
            factory.plot_dos(save=True, output_dir=output_dir, stacked=False)
            print("    [OK] phonon_dos_partial.png")
            plots_generated += 1

            factory.plot_dos(save=True, output_dir=output_dir, stacked=True)
            print("    [OK] phonon_dos_stacked.png")
            plots_generated += 1

            factory.plot_cumulative_dos(save=True, output_dir=output_dir)
            print("    [OK] phonon_dos_cumulative.png")
            plots_generated += 1

            # Hydrogen DOS if H present
            if 'H' in profile.elements:
                factory.plot_hydrogen_dos(save=True, output_dir=output_dir)
                print("    [OK] H_dos.png")
                plots_generated += 1

                factory.plot_h_mode_decomposition(save=True, output_dir=output_dir)
                print("    [OK] H_mode_decomposition.png")
                plots_generated += 1
        except Exception as e:
            print("    [FAIL] %s" % str(e)[:60])
    else:
        print("\n  [DOS] No DOS data available")

    # Thermodynamics
    if results.get('thermodynamics') and results['thermodynamics'].get('thermo_data'):
        print("\n  [Thermo] Generating thermodynamic figures...")
        try:
            factory.plot_thermodynamics(save=True, output_dir=output_dir)
            print("    [OK] phonon_thermodynamics.png")
            plots_generated += 1

            factory.plot_cv_dulong_petit(save=True, output_dir=output_dir)
            print("    [OK] cv_dulong_petit.png")
            plots_generated += 1

            factory.plot_free_energy_components(save=True, output_dir=output_dir)
            print("    [OK] free_energy_components.png")
            plots_generated += 1
        except Exception as e:
            print("    [FAIL] %s" % str(e)[:60])
    else:
        print("\n  [Thermo] No thermodynamic data available")

    # Force constants
    if results.get('ifc') or os.path.exists(os.path.join(workspace, 'FORCE_CONSTANTS')):
        print("\n  [IFC] Generating force constant figures...")
        try:
            # Try to extract IFC data
            fc_path = os.path.join(workspace, 'FORCE_CONSTANTS')
            if os.path.exists(fc_path):
                print("    [OK] FORCE_CONSTANTS loaded")
        except Exception as e:
            print("    [FAIL] %s" % str(e)[:60])

    # Thermal transport (if available)
    if results.get('transport'):
        print("\n  [Transport] Generating thermal transport figures...")
        try:
            trans = results['transport']
            if 'kappa_T' in trans:
                factory.plot_thermal_conductivity(
                    trans['temperatures'], trans['kappa_T'], save=True, output_dir=output_dir)
                print("    [OK] thermal_conductivity.png")
                plots_generated += 1
        except Exception as e:
            print("    [FAIL] %s" % str(e)[:60])

    # Interactive plots
    if not args.no_interactive:
        print("\n  [Interactive] Generating HTML interactive figures...")
        try:
            if results.get('bands_dos') and results['bands_dos'].get('band_dict'):
                fig = factory.interactive_plotter.plot_interactive_bands(
                    results['bands_dos']['band_dict'], formula=profile.formula)
                html_path = os.path.join(output_dir, 'interactive_band_structure.html')
                factory.interactive_plotter.save_interactive_html(fig, html_path)
                print("    [OK] interactive_band_structure.html")
                plots_generated += 1

            if results.get('thermodynamics') and results['thermodynamics'].get('thermo_data'):
                thermo = results['thermodynamics']['thermo_data']
                fig = factory.interactive_plotter.plot_interactive_thermo(
                    thermo['temperatures'], thermo['free_energy'],
                    thermo['entropy'], thermo['heat_capacity'], formula=profile.formula)
                html_path = os.path.join(output_dir, 'interactive_thermodynamics.html')
                factory.interactive_plotter.save_interactive_html(fig, html_path)
                print("    [OK] interactive_thermodynamics.html")
                plots_generated += 1

            if results.get('bands_dos') and results['bands_dos'].get('pdos_data'):
                freq, pdos, sym_to_idx = results['bands_dos']['pdos_data']
                fig = factory.interactive_plotter.plot_interactive_dos(
                    freq, pdos, sym_to_idx, formula=profile.formula)
                html_path = os.path.join(output_dir, 'interactive_dos.html')
                factory.interactive_plotter.save_interactive_html(fig, html_path)
                print("    [OK] interactive_dos.html")
                plots_generated += 1
        except Exception as e:
            print("    [FAIL] %s" % str(e)[:60])

    # Publication export
    if args.export_template == 'publication':
        print("\n  [Publication] Exporting publication package...")
        try:
            pub_dir = os.path.join(output_dir, 'publication')
            factory.pub_formatter.export_publication_package(
                figures={}, captions=[], output_dir=pub_dir,
                metadata={'formula': profile.formula, 'version': VERSION})
            print("    [OK] Publication package exported")
        except Exception as e:
            print("    [FAIL] %s" % str(e)[:60])

    # Summary
    print("\n" + "=" * 56)
    print("  VISUALIZATION COMPLETE")
    print("=" * 56)

    if plots_generated > 0:
        print(f"\n  Generated {plots_generated} figures in:")
        print(f"    {output_dir}\n")

        # List generated files
        files = sorted(os.listdir(output_dir))
        total_size = 0
        for fname in files:
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                total_size += size
                print(f"  {fname:<45} {size:>10d} bytes")

        print("-" * 56)
        print(f"  Total: {len(files)} files, {total_size/1024.0/1024.0:.1f} MB")
        print(f"\n  Open with: explorer {output_dir}")
    else:
        print("  No figures were generated. Check that postprocess completed successfully.")

    print("=" * 56)


def _load_existing_results(source, workspace, profile):
    """Load existing results from workspace or postprocess output.
    
    Tries multiple sources:
      1. Postprocess results directory (phonon_results_*)
      2. phonopy_disp.yaml + FORCE_CONSTANTS + band.yaml + thermo data
      3. Minimal results from band.yaml only
    
    Returns:
        dict: Results dictionary compatible with PhononFigureFactory, or None
    """
    import numpy as np
    
    results = {
        'profile': profile,
        'data_loader': None,
        'bands_dos': None,
        'thermodynamics': None,
        'hydrogen': {},
        'ifc': None,
        'group_velocities': {},
        'debye_waller': {},
        'mode_resolved_thermo': {}
    }

    # Mock phonon object for n_atoms
    from unittest.mock import Mock
    mock_phonon = Mock()
    mock_phonon.unitcell = Mock()
    mock_phonon.unitcell.numbers = list(range(getattr(profile, 'n_atoms', 1)))
    results['data_loader'] = {'phonon': mock_phonon}

    # Try to find postprocess output directory
    ws_dirs = []
    if os.path.isdir(workspace):
        for d in os.listdir(workspace):
            if d.startswith('phonon_results') and os.path.isdir(os.path.join(workspace, d)):
                ws_dirs.append(os.path.join(workspace, d))
    
    # Sort by modification time (most recent first)
    ws_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    results_dir = ws_dirs[0] if ws_dirs else None

    # Check for band.yaml
    band_path = None
    if results_dir:
        candidate = os.path.join(results_dir, 'band.yaml')
        if os.path.exists(candidate):
            band_path = candidate
    
    if not band_path and os.path.isdir(workspace):
        candidate = os.path.join(workspace, 'band.yaml')
        if os.path.exists(candidate):
            band_path = candidate

    # Parse band.yaml
    if band_path:
        try:
            import yaml
            with open(band_path, 'r') as f:
                band_data = yaml.safe_load(f)
            
            band_dict = _convert_band_yaml_to_dict(band_data)
            if band_dict:
                results['bands_dos'] = {
                    'band_dict': band_dict,
                    'min_freq': 0,
                    'stability': {'stable': True}
                }
                # Calculate min freq
                all_freqs = []
                for freq_arr in band_dict['frequencies']:
                    all_freqs.extend(freq_arr.flatten())
                if all_freqs:
                    results['bands_dos']['min_freq'] = min(all_freqs)
                    results['bands_dos']['stability']['stable'] = min(all_freqs) >= -0.1
        except Exception:
            pass

    # Check for thermodynamic_properties.dat
    thermo_path = None
    if results_dir:
        candidate = os.path.join(results_dir, 'thermodynamic_properties.dat')
        if os.path.exists(candidate):
            thermo_path = candidate
    
    if not thermo_path and os.path.isdir(workspace):
        # Look in phonon_results dirs
        for d in ws_dirs:
            candidate = os.path.join(d, 'thermodynamic_properties.dat')
            if os.path.exists(candidate):
                thermo_path = candidate
                break

    if thermo_path:
        try:
            temps, fe, ent, cv = [], [], [], []
            with open(thermo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('T'):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            temps.append(float(parts[0]))
                            fe.append(float(parts[1]))
                            ent.append(float(parts[2]))
                            cv.append(float(parts[3]))
                        except ValueError:
                            pass
            
            if temps:
                results['thermodynamics'] = {
                    'zpe': fe[0] if fe else 0,
                    'at_300K': {},
                    'validations': {},
                    'thermo_data': {
                        'temperatures': np.array(temps),
                        'free_energy': np.array(fe),
                        'entropy': np.array(ent),
                        'heat_capacity': np.array(cv)
                    }
                }
        except Exception:
            pass

    # Check for FORCE_CONSTANTS
    fc_path = None
    if results_dir:
        candidate = os.path.join(results_dir, 'FORCE_CONSTANTS')
        if os.path.exists(candidate):
            fc_path = candidate
    
    if not fc_path and os.path.isdir(workspace):
        candidate = os.path.join(workspace, 'FORCE_CONSTANTS')
        if os.path.exists(candidate):
            fc_path = candidate

    if fc_path:
        results['ifc'] = {'method': 'phonopy', 'path': fc_path}

    # Check if any data was loaded
    has_data = (results['bands_dos'] is not None or 
                results['thermodynamics'] is not None or
                results['ifc'] is not None)
    
    return results if has_data else None


def _convert_band_yaml_to_dict(band_data):
    """Convert phonopy band.yaml to band_dict format for visualization."""
    if not band_data:
        return None
    
    import numpy as np
    phonon = band_data.get('phonon', [])
    if not phonon:
        return None
    
    first = phonon[0]
    if isinstance(first, list):
        # Segmented format
        distances = []
        frequencies = []
        for segment in phonon:
            dists = [pt['distance'] for pt in segment]
            freqs = [[band['frequency'] for band in pt['band']] for pt in segment]
            distances.append(np.array(dists))
            frequencies.append(np.array(freqs))
    else:
        # Flat list - detect segments by distance discontinuities
        all_dists = [pt.get('distance', 0) for pt in phonon]
        all_freqs = [[band['frequency'] for band in pt['band']] for pt in phonon]
        
        segments_dist = []
        segments_freq = []
        current_dist = [all_dists[0]]
        current_freq = [all_freqs[0]]
        
        for i in range(1, len(all_dists)):
            if all_dists[i] < all_dists[i-1]:
                segments_dist.append(np.array(current_dist))
                segments_freq.append(np.array(current_freq))
                current_dist = [all_dists[i]]
                current_freq = [all_freqs[i]]
            else:
                current_dist.append(all_dists[i])
                current_freq.append(all_freqs[i])
        
        segments_dist.append(np.array(current_dist))
        segments_freq.append(np.array(current_freq))
        distances = segments_dist
        frequencies = segments_freq
    
    if distances:
        return {'distances': distances, 'frequencies': frequencies}
    return None


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    # No arguments → interactive wizard
    if len(sys.argv) == 1:
        print(BANNER)
        print()
        try:
            interactive_mode()
        except Exception as e:
            print(f"\n  [!] ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n  Press Enter to exit ...")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
        return

    # 'help' shortcut (no dashes)
    if sys.argv[1] in ('help', '--help', '-h'):
        print(BANNER)
        print(USAGE)
        return

    parser = argparse.ArgumentParser(
        prog="hydrophonokit",
        description=f"HydroPhonoKit v{VERSION} — Material-Aware Phonon Workflow Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE,
        add_help=True
    )

    parser.add_argument(
        'action',
        choices=['analyze', 'build', 'status', 'postprocess', 'visualize', 'elastic', 'qha', 'anharmonic', 'hstorage'],
        metavar='ACTION',
        help="analyze | build | status | postprocess | visualize | elastic | qha | anharmonic | hstorage"
    )
    parser.add_argument(
        '--source', '-s',
        metavar='DIR',
        default=None,
        help="Path to relaxed VASP directory (CONTCAR, INCAR, OUTCAR, POTCAR required)"
    )
    parser.add_argument(
        '--output', '-o',
        metavar='DIR',
        default=None,
        help="Output directory (required for 'build')"
    )
    parser.add_argument(
        '--workspace', '-w',
        metavar='DIR',
        default=None,
        help="Completed phonon workspace (required for 'postprocess')"
    )
    # H-storage specific arguments
    parser.add_argument(
        '--hydride',
        metavar='DIR',
        default=None,
        help="Hydride phase directory (for hstorage command)"
    )
    parser.add_argument(
        '--dehydride',
        metavar='DIR',
        default=None,
        help="Dehydride phase directory (for hstorage command)"
    )
    parser.add_argument(
        '--h2-energy',
        type=float,
        default=None,
        help="DFT energy of H2 molecule in eV (default: -6.77)"
    )
    parser.add_argument(
        '--n-h2',
        type=float,
        default=None,
        help="Moles of H2 released (auto-computed from n_h if not given)"
    )

    # HPC profile arguments (for build command)
    parser.add_argument(
        '--hpc',
        metavar='PROFILE',
        default=None,
        help="HPC profile: bridges2, roar, generic, custom, list"
    )
    parser.add_argument(
        '--hpc-config',
        metavar='FILE',
        default=None,
        help="YAML config file for custom HPC profile (use with --hpc custom)"
    )
    parser.add_argument(
        '--email',
        metavar='EMAIL',
        default=None,
        help="Email for SLURM job notifications"
    )

    # Postprocessor-specific arguments
    parser.add_argument(
        '--skip-phases',
        metavar='PHASES',
        default=None,
        help="Comma-separated list of phases to skip (e.g., 'hydrogen,plotting')"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help="Validate inputs without executing computation"
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=0,
        help="Maximum retries for failed phases (default: 0)"
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        default=False,
        help="Suppress verbose output"
    )
    parser.add_argument(
        '--export-template',
        metavar='TEMPLATE',
        default='full',
        choices=['minimal', 'full', 'publication'],
        help="Export template: minimal (key results), full (all data), publication (supplementary data)"
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        default=False,
        help="Disable caching of computed results"
    )
    parser.add_argument(
        '--cache-dir',
        metavar='DIR',
        default=None,
        help="Directory for cached files (default: .hydrophonokit_cache)"
    )
    parser.add_argument(
        '--memory-limit',
        type=float,
        default=0.0,
        help="Memory usage limit in GB (0 = no limit)"
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=0,
        help="Number of parallel jobs (0 = auto-detect)"
    )

    # Visualization-specific arguments
    parser.add_argument(
        '--figures',
        metavar='DIR',
        default=None,
        help="Output directory for figures (default: <workspace>/phonon_figures)"
    )
    parser.add_argument(
        '--theme',
        metavar='THEME',
        default='nature',
        choices=['nature', 'science', 'prl', 'acs', 'presentation', 'minimal'],
        help="Visualization theme theme (default: nature)"
    )
    parser.add_argument(
        '--format',
        metavar='FORMAT',
        default=None,
        help="Output format(s): png,pdf,eps,svg,tiff (comma-separated)"
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        default=False,
        help="Skip interactive HTML plot generation"
    )
    parser.add_argument(
        '--all-figures',
        action='store_true',
        default=False,
        help="Generate ALL possible figure types (bands, DOS, thermo, transport, etc.)"
    )

    args = parser.parse_args()
    print(BANNER)

    dispatch = {
        'analyze':     cmd_analyze,
        'build':       cmd_build,
        'status':      cmd_status,
        'postprocess': cmd_postprocess,
        'visualize':   cmd_visualize,
        'elastic':     cmd_elastic,
        'qha':         cmd_qha,
        'anharmonic':  cmd_anharmonic,
        'hstorage':    cmd_hstorage,
    }

    try:
        dispatch[args.action](args)
    except HydroPhonoKitError as e:
        print(f"\n  [!] ERROR: {e}")
        print("\n  Press Enter to exit ...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  [!] Interrupted by user. Exiting.")
        sys.exit(130)
    except Exception as e:
        print(f"\n  [!] UNEXPECTED ERROR: {e}")
        print("  This may be a bug in HydroPhonoKit. Please report it.")
        import traceback
        traceback.print_exc()
        print("\n  Press Enter to exit ...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(2)


if __name__ == '__main__':
    main()
