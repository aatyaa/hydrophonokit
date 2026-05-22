"""
=============================================================================
  HydroPhonoKit v2.2 -- Workspace Generator

  Generates complete VASP phonon calculation workspaces from a relaxed
  structure. Implements atomic workspace generation with rollback on failure.

  Generated structure:
    workspace/
    ├── 00_unitcell/POSCAR
    ├── 01_born/          (if insulator)
    ├── 02_displacements/
    │   ├── perfect/POSCAR
    │   ├── disp-001/
    │   ├── disp-002/
    │   └── ...
    ├── phonopy_disp.yaml
    ├── submit_born.sh
    ├── submit_all_disps.sh
    └── PHONON_PLAN.txt

  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
    [2] Parlinski et al., Phys. Rev. Lett. 78, 4063 (1997) -- Finite displacement
=============================================================================
"""
import os
import shutil
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import write_vasp
from .templates import get_incar_born, get_incar_force, get_kpoints_supercell, make_slurm_script
from .hpc_profiles import make_slurm_from_profile

# Version constant (single source of truth for this module)
VERSION = "2.7.0"


class GenerationError(Exception):
    """Raised when workspace generation fails."""
    pass


class PhononGenerator:
    """Generates VASP phonon workspace from a MaterialProfile.

    Implements atomic workspace generation: if generation fails partway,
    the entire output directory is cleaned up (rolled back) to prevent
    partial workspaces that could waste HPC compute time.
    """

    def __init__(self, unitcell, output_dir, profile, source_dir, hpc_profile=None):
        self.unitcell = unitcell
        self.output_dir = output_dir
        self.profile = profile
        self.source_dir = source_dir
        self.hpc_profile = hpc_profile
        self.n_disp = 0
        self._created_dirs = []  # Track for rollback

    def _rollback(self):
        """Clean up partially generated workspace on failure.

        Only removes directories that were created during this generation run.
        """
        if self._created_dirs:
            print(f"\n  [Generator] ⚠ Generation failed. Rolling back...")
            for dir_path in reversed(self._created_dirs):
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        print(f"    Cleaned up: {dir_path}")
                    except OSError as e:
                        print(f"    Warning: Could not clean {dir_path}: {e}")

    def _safe_makedirs(self, path):
        """Create directory and track for rollback."""
        os.makedirs(path, exist_ok=True)
        self._created_dirs.append(path)

    def generate(self):
        """Generate complete phonon calculation workspace.

        Returns:
            int: Number of displacement calculations generated

        Raises:
            GenerationError: If generation fails
        """
        p = self.profile
        try:
            self._safe_makedirs(self.output_dir)
            print(f"  [Generator] Output: {self.output_dir}")

            # Validate source POTCAR
            potcar_src = os.path.join(self.source_dir, 'POTCAR')
            potcar_exists = os.path.exists(potcar_src)
            if not potcar_exists:
                print(f"  [Generator] ⚠ WARNING: POTCAR not found in source directory!")
                print(f"      Source: {potcar_src}")
                print(f"      Generated workspace will be incomplete without POTCAR.")
                print(f"      VASP calculations will FAIL until POTCAR is copied manually.")

            # ---- 0. Unit cell reference ----
            uc_dir = os.path.join(self.output_dir, '00_unitcell')
            self._safe_makedirs(uc_dir)
            write_vasp(os.path.join(uc_dir, 'POSCAR'), self.unitcell)

            # ---- 1. Born directory (DFPT) — only if insulator ----
            if p.rec_born:
                born_dir = os.path.join(self.output_dir, '01_born')
                self._safe_makedirs(born_dir)
                write_vasp(os.path.join(born_dir, 'POSCAR'), self.unitcell)

                with open(os.path.join(born_dir, 'INCAR'), 'w') as f:
                    f.write(get_incar_born(
                        p.encut, p.kspacing,
                        ismear=p.rec_ismear, sigma=p.rec_sigma,
                        ivdw=p.rec_ivdw if p.rec_vdw else 0,
                        isym=p.rec_isym,
                        algo=getattr(p, 'rec_algo', 'Fast')))

                if potcar_exists:
                    shutil.copy2(potcar_src, os.path.join(born_dir, 'POTCAR'))
                else:
                    print(f"  [Generator] ⚠ Skipping POTCAR copy (not found in source)")

                with open(os.path.join(born_dir, 'run.sh'), 'w', newline='\n') as f:
                    f.write(self._make_run_sh(
                        f"{p.formula}_Born", 'born'))
                print(f"  [Generator] Born charges directory created (01_born/)")
            else:
                print(f"  [Generator] Born charges SKIPPED (metallic system)")

            # ---- 2. Phonopy Displacements ----
            dim = p.rec_supercell
            phonon = Phonopy(self.unitcell, np.diag(dim))
            phonon.generate_displacements(distance=p.rec_displacement)
            supercells = phonon.supercells_with_displacements
            self.n_disp = len(supercells)

            phonon.save(os.path.join(self.output_dir, 'phonopy_disp.yaml'))

            # ---- 3. Perfect supercell ----
            disp_base = os.path.join(self.output_dir, '02_displacements')
            perfect_dir = os.path.join(disp_base, 'perfect')
            self._safe_makedirs(perfect_dir)
            write_vasp(os.path.join(perfect_dir, 'POSCAR'), phonon.supercell)

            # ---- 4. Displacement directories ----
            for i, scell in enumerate(supercells):
                disp_id = f"{i+1:03d}"
                disp_dir = os.path.join(disp_base, f'disp-{disp_id}')
                self._safe_makedirs(disp_dir)

                write_vasp(os.path.join(disp_dir, 'POSCAR'), scell)
                with open(os.path.join(disp_dir, 'INCAR'), 'w') as f:
                    f.write(get_incar_force(
                        p.encut, kspacing=p.kspacing,
                        ismear=p.rec_ismear, sigma=p.rec_sigma,
                        ivdw=p.rec_ivdw if p.rec_vdw else 0,
                        isym=p.rec_isym,
                        algo=getattr(p, 'rec_algo', 'Fast')))
                with open(os.path.join(disp_dir, 'KPOINTS'), 'w') as f:
                    f.write(get_kpoints_supercell(p.rec_kpoints_sc))
                if potcar_exists:
                    shutil.copy2(potcar_src, os.path.join(disp_dir, 'POTCAR'))
                with open(os.path.join(disp_dir, 'run.sh'), 'w', newline='\n') as f:
                    f.write(self._make_run_sh(
                        f"{p.formula}_P{disp_id}", 'disp'))

            # ---- 5. Master submission scripts ----
            with open(os.path.join(self.output_dir, 'submit_born.sh'), 'w', newline='\n') as f:
                f.write("#!/bin/bash\n")
                if p.rec_born:
                    f.write("cd 01_born\nsbatch run.sh\ncd ..\n")
                    f.write('echo "Born charges job submitted!"\n')
                else:
                    f.write('echo "Born charges not required for this system."\n')

            with open(os.path.join(self.output_dir, 'submit_all_disps.sh'), 'w', newline='\n') as f:
                f.write("#!/bin/bash\ncd 02_displacements\n")
                f.write("count=0\n")
                f.write("for dir in disp-*/; do\n")
                f.write('    cd "$dir"\n    sbatch run.sh\n    cd ..\n')
                f.write('    count=$((count + 1))\n')
                f.write("done\ncd ..\n")
                f.write('echo "Submitted $count displacement jobs!"\n')

            # ---- 6. Summary report ----
            self._write_report()

            print(f"  [Generator] Created {self.n_disp} displacement directories.")
            return self.n_disp

        except Exception as e:
            self._rollback()
            raise GenerationError(
                f"Workspace generation failed: {e}\n"
                f"Partial workspace has been cleaned up (rolled back)."
            ) from e

    def _make_run_sh(self, job_name, calc_type='disp'):
        """Generate run.sh content, using HPC profile if available.

        Args:
            job_name: SLURM job name
            calc_type: 'born' or 'disp' (determines walltime)

        Returns:
            str: SLURM script content
        """
        if self.hpc_profile:
            walltime = (self.hpc_profile['born_walltime'] if calc_type == 'born'
                        else self.hpc_profile['disp_walltime'])
            return make_slurm_from_profile(
                self.hpc_profile, job_name, walltime,
                job_prefix=f"HydroPhonoKit {calc_type} calculation"
            )
        else:
            # Legacy fallback (backward compatible)
            p = self.profile
            hours = (p.rec_born_time_hrs if calc_type == 'born'
                     else p.rec_disp_time_hrs)
            return make_slurm_script(job_name, hours)

    def _write_report(self):
        """Write a PHONON_PLAN.txt summary inside the workspace."""
        p = self.profile
        sc = p.rec_supercell
        report_path = os.path.join(self.output_dir, 'PHONON_PLAN.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"  HydroPhonoKit v{VERSION} - Calculation Plan Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"  Material:       {p.formula}\n")
            f.write(f"  Space Group:    {p.space_group}\n")
            f.write(f"  Electronic:     {'Insulator' if p.is_insulator else 'Metal'}\n")
            f.write(f"  ENCUT:          {p.encut} eV\n")
            f.write(f"  Supercell:      {sc[0]}x{sc[1]}x{sc[2]} ({p.sc_atoms} atoms)\n")
            f.write(f"  Displacement:   {p.rec_displacement} A\n")
            f.write(f"  Born charges:   {'Yes' if p.rec_born else 'No'}\n")
            f.write(f"  vdW:            {'IVDW=' + str(p.rec_ivdw) if p.rec_vdw else 'None'}\n")
            f.write(f"  ALGO:           {getattr(p, 'rec_algo', 'Fast')}\n")
            isym_val = getattr(p, 'rec_isym', None)
            f.write(f"  ISYM:           {isym_val if isym_val is not None else 'default'}\n")
            f.write(f"  Displacements:  {self.n_disp}\n")
            if self.hpc_profile:
                f.write(f"\n  HPC Profile:    {self.hpc_profile['name']}\n")
                f.write(f"  Cores/node:     {self.hpc_profile['cores_per_node']}\n")
                f.write(f"  Partition:      {self.hpc_profile['partition']}\n")
                f.write(f"  Born walltime:  {self.hpc_profile['born_walltime']}\n")
                f.write(f"  Disp walltime:  {self.hpc_profile['disp_walltime']}\n")
            else:
                f.write(f"\n  HPC Profile:    generic (use --hpc to customize)\n")
            f.write("\n")
            if p.warnings:
                f.write("  WARNINGS:\n")
                for w in p.warnings:
                    f.write(f"  - {w}\n")
            f.write("\n" + "=" * 60 + "\n")
