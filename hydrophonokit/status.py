"""
=============================================================================
  HydroPhonoKit v2.7 -- Workspace Status Monitor

  Scans a generated workspace and reports on the VASP calculation status
  for Born charges and all displacements.

  Identifies:
    - NOT_STARTED: No OUTCAR found
    - RUNNING: OUTCAR exists but incomplete
    - CONVERGED: Electronic SCF converged successfully
    - FAILED_SCF: Finished but SCF did not converge (reached NELM)
=============================================================================
"""
import os
import glob
from collections import Counter


def check_vasp_static_outcar(outcar_path):
    """Check a static VASP run (IBRION=-1) OUTCAR for completion and SCF convergence.
    
    Returns:
        status (str): 'NOT_STARTED', 'RUNNING', 'CONVERGED', 'FAILED_SCF', 'UNKNOWN_ERROR'
        energy (float or None): Final energy if available
    """
    if not os.path.exists(outcar_path):
        return 'NOT_STARTED', None

    try:
        # Read the last 200 lines to quickly find completion flags
        with open(outcar_path, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            chunk_size = min(16384, file_size)
            f.seek(file_size - chunk_size)
            lines = f.read().decode('utf-8', errors='ignore').splitlines()

        is_completed = False
        scf_converged = False
        energy = None

        for line in reversed(lines):
            if "General timing and accounting" in line or "Voluntary context switches" in line:
                is_completed = True
            if "aborting loop because EDIFF is reached" in line:
                scf_converged = True
            if energy is None and "free  energy   TOTEN" in line:
                try:
                    energy = float(line.split()[-2])
                except (ValueError, IndexError):
                    pass

        if not is_completed:
            # Check if it just started or is still running
            return 'RUNNING', energy

        if scf_converged:
            return 'CONVERGED', energy
        else:
            return 'FAILED_SCF', energy

    except Exception:
        return 'UNKNOWN_ERROR', None


def scan_workspace(workspace_dir):
    """Scan the workspace and return a structured status report."""
    report = {
        'born': {'required': False, 'status': None, 'energy': None},
        'displacements': {
            'total': 0,
            'completed': 0,
            'converged': 0,
            'running': 0,
            'failed_scf': 0,
            'not_started': 0,
            'unknown': 0,
            'details': {}
        },
        'ready_for_postprocessing': False
    }

    # 1. Check Born charges
    born_dir = os.path.join(workspace_dir, '01_born')
    if os.path.exists(born_dir):
        report['born']['required'] = True
        status, energy = check_vasp_static_outcar(os.path.join(born_dir, 'OUTCAR'))
        report['born']['status'] = status
        report['born']['energy'] = energy

    # 2. Check Displacements
    disp_base = os.path.join(workspace_dir, '02_displacements')
    if not os.path.exists(disp_base):
        return report

    disp_dirs = sorted(glob.glob(os.path.join(disp_base, "disp-*")))
    report['displacements']['total'] = len(disp_dirs)

    counts = Counter()
    for disp in disp_dirs:
        disp_id = os.path.basename(disp)
        status, _ = check_vasp_static_outcar(os.path.join(disp, 'OUTCAR'))
        report['displacements']['details'][disp_id] = status
        counts[status] += 1

    report['displacements']['not_started'] = counts['NOT_STARTED']
    report['displacements']['running'] = counts['RUNNING']
    report['displacements']['converged'] = counts['CONVERGED']
    report['displacements']['failed_scf'] = counts['FAILED_SCF']
    report['displacements']['unknown'] = counts['UNKNOWN_ERROR']
    report['displacements']['completed'] = counts['CONVERGED'] + counts['FAILED_SCF'] + counts['UNKNOWN_ERROR']

    # 3. Determine readiness
    disp_ready = report['displacements']['converged'] == report['displacements']['total'] and report['displacements']['total'] > 0
    born_ready = not report['born']['required'] or report['born']['status'] == 'CONVERGED'
    
    report['ready_for_postprocessing'] = disp_ready and born_ready

    return report


def print_status_report(workspace_dir):
    """Print a formatted status report to the console."""
    if not os.path.exists(workspace_dir):
        print(f"\n  [!] Workspace not found: {workspace_dir}")
        return

    print("\n" + "=" * 60)
    print(f"  HydroPhonoKit Workspace Status")
    print(f"  Location: {workspace_dir}")
    print("=" * 60)

    report = scan_workspace(workspace_dir)

    # Born Status
    if report['born']['required']:
        st = report['born']['status']
        en = report['born']['energy']
        sym = "✅" if st == "CONVERGED" else "⏳" if st == "RUNNING" else "❌" if st == "FAILED_SCF" else "⚫"
        en_str = f" (E={en:.4f} eV)" if en is not None else ""
        print(f"  Born charges:    {sym} {st}{en_str}")
    else:
        print(f"  Born charges:    Not required")

    print("-" * 60)

    # Displacements Status
    disp = report['displacements']
    tot = disp['total']
    comp = disp['completed']
    print(f"  Displacements:   {comp}/{tot} completed")
    print(f"    ✅ Converged:   {disp['converged']}")
    
    if disp['running'] > 0:
        print(f"    ⏳ Running:     {disp['running']}")
        running_ids = [k for k, v in disp['details'].items() if v == 'RUNNING']
        if running_ids:
            print(f"                    ({', '.join(running_ids[:5])}{'...' if len(running_ids) > 5 else ''})")
            
    if disp['failed_scf'] > 0:
        print(f"    ❌ Failed SCF:  {disp['failed_scf']}")
        failed_ids = [k for k, v in disp['details'].items() if v == 'FAILED_SCF']
        if failed_ids:
            print(f"                    ({', '.join(failed_ids[:5])}{'...' if len(failed_ids) > 5 else ''})")
            
    if disp['not_started'] > 0:
        print(f"    ⚫ Not started: {disp['not_started']}")

    print("-" * 60)
    
    if report['ready_for_postprocessing']:
        print("  Readiness:       ✅ READY FOR POSTPROCESSING")
    else:
        print("  Readiness:       ❌ NOT READY")
        if disp['failed_scf'] > 0:
            print("                   (Warning: Some jobs failed SCF convergence. Please check ALGO/AMIX)")
        
    print("=" * 60 + "\n")
