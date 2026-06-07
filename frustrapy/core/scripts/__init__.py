"""Out-of-process toolchain assets for FrustraPy.

This package holds the LIVE shell + Perl + precompiled-binary toolchain that the
analysis engine invokes via ``subprocess`` (e.g. ``AWSEMFiles/AWSEMTools/
PdbCoords2Lammps.sh``, ``GenerateVisualizations.pl``, ``GenerateChargeFile.pl``,
and the ``lmp_serial_*`` binaries). It is intentionally NOT a Python import
facade: the pure-Python rewrites of those steps were dead code and were removed
in the Phase 3 dead-code purge. The canonical Python implementations live under
``frustrapy.utils`` (e.g. ``renum_files`` at ``frustrapy/utils/helpers.py``).
"""
