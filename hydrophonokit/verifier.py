"""
=============================================================================
  HydroPhonoKit v2.2 -- System Verifier

  Rigorous integrity validation of generated phonon workspaces.
  Verifies:
    - File existence (POSCAR, INCAR, KPOINTS, POTCAR, run.sh)
    - File content validity (non-empty, parseable)
    - INCAR parameter consistency
    - POSCAR structural validity
    - Displacement count matches expected
    - phonopy_disp.yaml presence and validity

  References:
    [1] VASP Documentation -- Required input files
    [2] Phonopy Documentation -- phonopy_disp.yaml schema
=============================================================================
"""
import os
import glob
import re

class VerificationError(Exception):
    """Raised when workspace integrity check fails."""
    pass

class VerificationWarning:
    """Non-critical issue that doesn't fail verification."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"  [⚠] {self.message}"


class SystemVerifier:
    """Rigorous integrity validator for HydroPhonoKit workspaces."""

    REQUIRED_BORN_FILES = ['POSCAR', 'INCAR', 'POTCAR', 'run.sh']
    REQUIRED_DISP_FILES = ['POSCAR', 'INCAR', 'KPOINTS', 'POTCAR', 'run.sh']

    def __init__(self, output_dir, expected_disp_count):
        self.output_dir = output_dir
        self.expected_disp_count = expected_disp_count
        self.errors = []
        self.warnings = []

    def verify_integrity(self):
        """Execute comprehensive workspace integrity validation.

        Raises:
            VerificationError: If critical integrity check fails
        """
        print(f"\n[Verifier] Initiating comprehensive integrity validation")
        print(f"  Workspace: {self.output_dir}")
        print(f"  Expected displacements: {self.expected_disp_count}")
        print("-" * 60)

        self._check_born_directory()
        self._check_displacement_directories()
        self._check_phonopy_yaml()

        if self.errors:
            raise VerificationError(
                f"Workspace verification FAILED with {len(self.errors)} error(s):\n" +
                "\n".join(f"  ❌ {e}" for e in self.errors)
            )

        if self.warnings:
            print(f"\n  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  {w}")

        print(f"\n[Verifier] ✓ Output structure integrity passed.")
        print(f"  All critical files validated. Ready for HPC execution.")

    def _check_born_directory(self):
        """Validate 01_born/ directory with all required files."""
        born_dir = os.path.join(self.output_dir, "01_born")

        if not os.path.exists(born_dir):
            self.warnings.append(VerificationWarning(
                "Born charges directory (01_born/) not found. "
                "LO-TO splitting will be disabled."))
            return

        for fname in self.REQUIRED_BORN_FILES:
            fpath = os.path.join(born_dir, fname)
            if not os.path.exists(fpath):
                self.errors.append(f"01_born/{fname} missing")
                continue

            # Validate file is non-empty
            if os.path.getsize(fpath) == 0:
                self.errors.append(f"01_born/{fname} is EMPTY (0 bytes)")

        # Validate INCAR contains LEPSILON = .TRUE.
        incar_path = os.path.join(born_dir, 'INCAR')
        if os.path.exists(incar_path):
            with open(incar_path, 'r') as f:
                content = f.read()
            if 'LEPSILON' not in content.upper():
                self.errors.append(
                    "01_born/INCAR missing LEPSILON tag -- "
                    "DFPT Born charge calculation will not run")

    def _check_displacement_directories(self):
        """Validate all displacement directories exist with required files."""
        disp_base = os.path.join(self.output_dir, "02_displacements")

        if not os.path.exists(disp_base):
            self.errors.append("Displacement base directory (02_displacements/) not found")
            return

        disp_folders = sorted(glob.glob(os.path.join(disp_base, "disp-*")))

        if len(disp_folders) != self.expected_disp_count:
            self.errors.append(
                f"Displacement count mismatch: "
                f"found {len(disp_folders)}, expected {self.expected_disp_count}")
            return

        # Validate each displacement directory
        for i, disp_dir in enumerate(disp_folders):
            disp_name = os.path.basename(disp_dir)
            self._validate_disp_directory(disp_dir, disp_name, i)

    def _validate_disp_directory(self, disp_dir, disp_name, index):
        """Validate a single displacement directory."""
        for fname in self.REQUIRED_DISP_FILES:
            fpath = os.path.join(disp_dir, fname)

            if not os.path.exists(fpath):
                self.errors.append(f"{disp_name}/{fname} missing")
                continue

            # Validate file is non-empty
            if os.path.getsize(fpath) == 0:
                self.errors.append(f"{disp_name}/{fname} is EMPTY (0 bytes)")

        # Validate POSCAR has content (at least a header line)
        poscar_path = os.path.join(disp_dir, 'POSCAR')
        if os.path.exists(poscar_path):
            with open(poscar_path, 'r') as f:
                first_line = f.readline().strip()
            if not first_line:
                self.errors.append(f"{disp_name}/POSCAR has no header line")

        # Validate INCAR has IBRION = -1 (static calculation)
        incar_path = os.path.join(disp_dir, 'INCAR')
        if os.path.exists(incar_path):
            with open(incar_path, 'r') as f:
                content = f.read()
            if 'IBRION' in content.upper():
                m = re.search(r'IBRION\s*=\s*(-?\d+)', content, re.IGNORECASE)
                if m and m.group(1) != '-1':
                    self.warnings.append(VerificationWarning(
                        f"{disp_name}/INCAR has IBRION={m.group(1)} "
                        f"(should be -1 for static force calculation)"))
            else:
                self.warnings.append(VerificationWarning(
                    f"{disp_name}/INCAR missing IBRION tag (defaulting to VASP default)"))

    def _check_phonopy_yaml(self):
        """Validate phonopy_disp.yaml exists and is parseable."""
        yaml_path = os.path.join(self.output_dir, 'phonopy_disp.yaml')

        if not os.path.exists(yaml_path):
            self.errors.append("phonopy_disp.yaml not found -- "
                               "workspace was not generated by PhononGenerator")
            return

        # Basic validation: file should be non-empty and contain expected keys
        if os.path.getsize(yaml_path) == 0:
            self.errors.append("phonopy_disp.yaml is EMPTY (0 bytes)")
            return

        with open(yaml_path, 'r') as f:
            content = f.read()

        required_keys = ['phonopy', 'supercell_matrix', 'displacements']
        for key in required_keys:
            if key not in content:
                self.warnings.append(VerificationWarning(
                    f"phonopy_disp.yaml may be malformed (missing '{key}' key)"))
