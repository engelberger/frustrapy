"""Unit guards for the frustration classification thresholds and the classifiers
that consume them.

These cutoffs are scientifically load-bearing and historically under-documented
(they were audited one-by-one against ``frustratometeR``'s R source). A silent
change to any of them would mis-classify every contact/residue while the energies
still look reasonable, so this module pins:

  * the exact constant values in ``frustrapy.core.constants`` (the single source
    of truth imported everywhere);
  * that the table classifier (``utils/helpers.py``) and the visualization layer
    reference those same constant objects rather than re-hardcoding the numbers;
  * the 3D contact-mutation classifier boundaries, including the ``[0.78, 1.0)``
    case that the incomplete 399b975 fix used to mislabel as neutral.

All tests here are fast (no LAMMPS / no subprocess).
"""

from frustrapy.core import constants
from frustrapy.visualization.structure import classify_contact_frustration


def test_contact_cutoff_constants_exact():
    """Configurational/mutational contact class cutoffs — match frustratometeR."""
    assert constants.FRST_HIGHLY_MAX == -1.0
    assert constants.FRST_MINIMALLY_MIN_CONTACT == 0.78


def test_singleres_and_welltype_constants_exact():
    """Single-residue (plot-only) and well-type geometry cutoffs.

    The 0.58 single-residue cutoff is INTENTIONALLY distinct from the 0.78 contact
    cutoff (frustratometeR uses 0.78 for contact tables, 0.58 for the per-residue
    plot) — they must never be collapsed.
    """
    assert constants.FRST_MINIMALLY_MIN_SINGLERES == 0.58
    assert constants.FRST_MINIMALLY_MIN_SINGLERES != constants.FRST_MINIMALLY_MIN_CONTACT
    assert constants.WELLTYPE_DISTANCE_CUTOFF == 6.5
    assert constants.WATER_MEDIATED_DENSITY_CUTOFF == 2.6


def test_constants_are_the_single_source_of_truth():
    """The table classifier and the visualization layer import the constants
    (no re-hardcoded magic numbers that could drift)."""
    from frustrapy.utils import helpers
    from frustrapy.visualization import plots, structure

    for module in (helpers, plots, structure):
        assert module.FRST_HIGHLY_MAX is constants.FRST_HIGHLY_MAX
        assert module.FRST_MINIMALLY_MIN_CONTACT is constants.FRST_MINIMALLY_MIN_CONTACT


def test_classify_contact_frustration_boundaries():
    """The 3D contact-mutation classifier uses the asymmetric contact cutoffs.

    Boundaries are inclusive at -1.0 (highly) and 0.78 (minimally), matching the
    table classifier and frustratometeR's visualization.R:618-620.
    """
    assert classify_contact_frustration(-2.0) == "highly"
    assert classify_contact_frustration(-1.0) == "highly"  # inclusive
    assert classify_contact_frustration(-0.99) == "neutral"
    assert classify_contact_frustration(0.0) == "neutral"
    assert classify_contact_frustration(0.77) == "neutral"
    assert classify_contact_frustration(0.78) == "minimally"  # inclusive
    assert classify_contact_frustration(2.0) == "minimally"


def test_classify_contact_frustration_399b975_regression():
    """Regression for the incomplete 3D-view fix: a contact with FrstIndex in
    ``[0.78, 1.0)`` is minimally frustrated. The old symmetric ``+-delta_threshold``
    (default 1.0) classified it as neutral (and the label/cartoon/legend kept doing
    so even after the cylinder path was fixed)."""
    assert classify_contact_frustration(0.785) == "minimally"
    assert classify_contact_frustration(0.99) == "minimally"


def test_classify_matches_table_classifier_bins():
    """The 3D-view classifier returns the same class the output-table classifier
    (helpers.renum_files) assigns, across the documented bins."""
    hi = constants.FRST_HIGHLY_MAX
    lo = constants.FRST_MINIMALLY_MIN_CONTACT

    def table_class(frst):
        # Mirror utils/helpers.py:260-267 (the FrstState column logic).
        if frst <= hi:
            return "highly"
        if hi < frst < lo:
            return "neutral"
        return "minimally"

    for frst in (-3.0, -1.0, -0.5, 0.0, 0.5, 0.77, 0.78, 1.0, 3.5):
        assert classify_contact_frustration(frst) == table_class(frst)
