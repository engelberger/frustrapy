"""In-container unit tests for the atomic-backend mode -> decoy-strategy mapping
(AA-MODES). No Rosetta: the Rosetta scoring step is injected as a mockable callable,
so every test here exercises the mode LOGIC (which positions each mode is allowed to
re-identify, decoy counts, seed reproducibility, the shared pooling) and the routing
of each mode through the shared post-processor into the right table shape.

Honesty framing under test (the mission's central requirement):

* ``configurational`` is the ONE parity-backed atomic mode (the reference permutation
  scheme); its ``parity_status`` is ``parity-backed``.
* ``mutational`` and ``singleresidue`` are EXTENSIONS BEYOND THE PAPER; their
  ``parity_status`` is ``experimental``. These tests check their decoy-generation
  logic and table routing, NOT numerical parity (there is no oracle for them).

The decoy strategies and the shared aggregation/pooling are pure Python and need no
reference data, so this whole file runs in-container with no skips.
"""

import numpy as np
import pytest

from frustrapy.backends import atomic_modes as am
from frustrapy.backends import atomic_post as ap


# ---------------------------------------------------------------------------
# Mode metadata + the parity-vs-extension honesty contract.
# ---------------------------------------------------------------------------

def test_only_configurational_is_parity_backed():
    """Exactly one atomic mode is parity-backed (configurational, the reference
    permutation scheme); the other two are clearly-labeled extensions."""
    assert set(am.MODES) == {"configurational", "mutational", "singleresidue"}
    assert am.MODES["configurational"].parity_status == am.PARITY_BACKED
    assert am.MODES["configurational"].is_parity_backed
    for ext in ("mutational", "singleresidue"):
        assert am.MODES[ext].parity_status == am.EXPERIMENTAL
        assert am.MODES[ext].is_experimental
        assert am.is_experimental_mode(ext)
    assert not am.is_experimental_mode("configurational")


def test_mode_table_routing_metadata():
    """Each mode declares the downstream table shape it routes to (contacts vs
    single-residue)."""
    assert "14-column" in am.MODES["configurational"].table
    assert "14-column" in am.MODES["mutational"].table
    assert "8-column" in am.MODES["singleresidue"].table


def test_mode_info_unknown_raises():
    with pytest.raises(ValueError):
        am.mode_info("nope")
    with pytest.raises(ValueError):
        am.get_decoy_strategy("nope")


def test_strategy_reports_its_parity_status():
    assert am.get_decoy_strategy("configurational").parity_status == am.PARITY_BACKED
    assert am.get_decoy_strategy("mutational").parity_status == am.EXPERIMENTAL
    assert am.get_decoy_strategy("singleresidue").parity_status == am.EXPERIMENTAL


# ---------------------------------------------------------------------------
# Configurational decoy strategy (PARITY-BACKED: whole-sequence permutation).
# ---------------------------------------------------------------------------

NATIVE = "DIQVQVNIDDNGKNFDYTYTVTTESELQK"


def test_configurational_is_whole_sequence_permutation():
    strat = am.get_decoy_strategy("configurational")
    groups = strat.generate(NATIVE, n_decoys=6, seed=1)
    # One protein-wide group (key=None): the reference pools one statistic over all.
    assert len(groups) == 1
    assert groups[0].key is None
    specs = groups[0].specs
    assert len(specs) == 6
    for spec in specs:
        # Composition preserved (a permutation) and the whole sequence is the domain.
        assert sorted(spec.sequence) == sorted(NATIVE)
        assert spec.varied_positions == tuple(range(len(NATIVE)))
        assert am.spec_respects_domain(NATIVE, spec)


def test_configurational_seed_reproducible():
    strat = am.get_decoy_strategy("configurational")
    a = strat.generate(NATIVE, n_decoys=4, seed=7)
    b = strat.generate(NATIVE, n_decoys=4, seed=7)
    c = strat.generate(NATIVE, n_decoys=4, seed=8)
    seqs = lambda g: [s.sequence for s in g[0].specs]
    assert seqs(a) == seqs(b)
    assert seqs(a) != seqs(c)


# ---------------------------------------------------------------------------
# Single-residue decoy strategy (EXPERIMENTAL: re-identify only site i).
# ---------------------------------------------------------------------------

def test_singleresidue_domain_is_one_site():
    strat = am.get_decoy_strategy("singleresidue")
    groups = strat.generate("ACDEFG", sites=[1, 4])
    assert [g.key for g in groups] == [1, 4]
    for g in groups:
        i = g.key
        # Exhaustive 20-identity scan by default.
        assert len(g.specs) == len(am.ATOMIC_ALPHABET) == 20
        for spec in g.specs:
            assert spec.varied_positions == (i,)
            # Only site i may differ from native; every other position is native.
            assert am.spec_respects_domain("ACDEFG", spec)
            diffs = am.positions_differing_from_native("ACDEFG", spec.sequence)
            assert set(diffs).issubset({i})
        # The scan re-identifies site i to every alphabet identity exactly once.
        assert sorted(spec.sequence[i] for spec in g.specs) == sorted(am.ATOMIC_ALPHABET)


def test_singleresidue_defaults_to_all_sites():
    strat = am.get_decoy_strategy("singleresidue")
    groups = strat.generate("ACDE")
    assert [g.key for g in groups] == [0, 1, 2, 3]


def test_singleresidue_sampled_is_seeded():
    strat = am.get_decoy_strategy("singleresidue")
    a = strat.generate("ACDEFG", sites=[2], n_decoys_per_site=5, seed=3)
    b = strat.generate("ACDEFG", sites=[2], n_decoys_per_site=5, seed=3)
    c = strat.generate("ACDEFG", sites=[2], n_decoys_per_site=5, seed=4)
    assert len(a[0].specs) == 5
    seqs = lambda g: [s.sequence for s in g[0].specs]
    assert seqs(a) == seqs(b)
    assert seqs(a) != seqs(c)


def test_singleresidue_out_of_range_raises():
    strat = am.get_decoy_strategy("singleresidue")
    with pytest.raises(ValueError):
        strat.generate("ACDE", sites=[9])


# ---------------------------------------------------------------------------
# Mutational decoy strategy (EXPERIMENTAL: re-identify only the contact pair i,j).
# ---------------------------------------------------------------------------

def test_mutational_domain_is_the_contact_pair():
    strat = am.get_decoy_strategy("mutational")
    groups = strat.generate("ACDEFG", contacts=[(0, 5), (1, 3)], n_decoys_per_contact=8, seed=1)
    assert [g.key for g in groups] == [(0, 5), (1, 3)]
    for g in groups:
        i, j = g.key
        assert len(g.specs) == 8
        for spec in g.specs:
            assert spec.varied_positions == (i, j)
            # Only positions i and j may differ from native.
            diffs = am.positions_differing_from_native("ACDEFG", spec.sequence)
            assert set(diffs).issubset({i, j})
            assert am.spec_respects_domain("ACDEFG", spec)


def test_mutational_exhaustive_pair_grid():
    strat = am.get_decoy_strategy("mutational")
    groups = strat.generate("ACDEFG", contacts=[(0, 5)])
    # Default = exhaustive identity-pair grid over the two sites.
    assert len(groups[0].specs) == len(am.ATOMIC_ALPHABET) ** 2 == 400


def test_mutational_sampled_is_seeded():
    strat = am.get_decoy_strategy("mutational")
    a = strat.generate("ACDEFG", contacts=[(0, 5)], n_decoys_per_contact=10, seed=5)
    b = strat.generate("ACDEFG", contacts=[(0, 5)], n_decoys_per_contact=10, seed=5)
    c = strat.generate("ACDEFG", contacts=[(0, 5)], n_decoys_per_contact=10, seed=6)
    seqs = lambda g: [s.sequence for s in g[0].specs]
    assert seqs(a) == seqs(b)
    assert seqs(a) != seqs(c)


def test_mutational_requires_contacts():
    strat = am.get_decoy_strategy("mutational")
    with pytest.raises(ValueError):
        strat.generate("ACDE")  # contacts is mandatory
    with pytest.raises(ValueError):
        strat.generate("ACDE", contacts=[(0, 9)])  # out of range


# ---------------------------------------------------------------------------
# Shared aggregation seam: one scoring callable, per-group pooling.
# ---------------------------------------------------------------------------

def test_pooled_statistics_drops_zeros_population_std():
    mean, std, n = am.pooled_statistics([2.0, 0.0, 4.0, 0.0, 6.0])
    assert n == 3  # zeros dropped (the reference's temp != 0 test)
    assert mean == pytest.approx(4.0)
    assert std == pytest.approx(np.std([2.0, 4.0, 6.0]))  # ddof=0
    # All-zero pool -> NaN/NaN/0 (no usable decoys), matching the reference.
    m, s, k = am.pooled_statistics([0.0, 0.0])
    assert k == 0 and np.isnan(m) and np.isnan(s)


def test_aggregate_groups_uses_one_shared_score_fn_across_modes():
    """The SAME scoring callable pools every mode's groups; only the decoy specs
    (and hence the group keys) differ. This is the 'parameterize only the decoy
    step' contract."""
    calls = []

    def score_fn(spec, group):
        calls.append((group.key, spec.sequence))
        # A deterministic surrogate energy: length-weighted hash-free function of the
        # varied identities, so groups get distinct, reproducible pools.
        return float(sum(ord(spec.sequence[p]) for p in spec.varied_positions))

    seq = "ACDEFG"
    cfg = am.get_decoy_strategy("configurational").generate(seq, n_decoys=3, seed=1)
    sr = am.get_decoy_strategy("singleresidue").generate(seq, sites=[0, 2])
    mut = am.get_decoy_strategy("mutational").generate(
        seq, contacts=[(0, 5)], n_decoys_per_contact=4, seed=1
    )

    cfg_stat = am.aggregate_groups(cfg, score_fn)
    assert set(cfg_stat) == {None}
    sr_stat = am.aggregate_groups(sr, score_fn)
    assert set(sr_stat) == {0, 2}
    mut_stat = am.aggregate_groups(mut, score_fn)
    assert set(mut_stat) == {(0, 5)}

    # The same callable was invoked for all three mode shapes.
    assert {k for (k, _seq) in calls} == {None, 0, 2, (0, 5)}
    # Each statistic is a (mean, std, n) triple.
    for stat in (*cfg_stat.values(), *sr_stat.values(), *mut_stat.values()):
        assert len(stat) == 3


# ---------------------------------------------------------------------------
# Routing each mode through the SHARED post-processor to the right table shape.
# Energies are mocked (no Rosetta); the point is the table SHAPE, not the numbers.
# ---------------------------------------------------------------------------

def _toy_geometry():
    """Six residues on one chain, rep atoms 4 A apart in a line."""
    coords = np.array([[4.0 * k, 0.0, 0.0] for k in range(6)], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    return ap.ContactGeometry(
        cid_list=tuple(f"A{k + 1}" for k in range(6)),
        aa=tuple("ACDEFG"),
        rep_coords=coords,
        dist=dist,
    )


def _write_equivalences(job, geom):
    with open(job / "native.pdb_equivalences.txt", "w") as f:
        for k, cid in enumerate(geom.cid_list):
            f.write(f"{cid[0]} {k + 1} {cid[1:]}\n")


CONTACT_COLS = [
    "Res1", "Res2", "ChainRes1", "ChainRes2", "DensityRes1", "DensityRes2",
    "AA1", "AA2", "NativeEnergy", "DecoyEnergy", "SDEnergy", "FrstIndex",
    "Welltype", "FrstState",
]
SINGLERES_COLS = [
    "Res", "ChainRes", "DensityRes", "AA", "NativeEnergy", "DecoyEnergy",
    "SDEnergy", "FrstIndex",
]


def test_configurational_routes_to_14col_contact_table(tmp_path):
    """Configurational: protein-wide engine result -> shared contact writer ->
    shared process_results -> canonical 14-column contact table."""
    import pandas as pd
    from frustrapy.utils.helpers import renum_files
    from frustrapy.backends import atomic_engine as ae

    geom = _toy_geometry()
    # Synthetic engine result (stands in for the maintainer Rosetta run).
    engine = ae.EngineResult(
        native_residue_energy={f"A{k + 1}": -1.0 * (k + 1) for k in range(6)},
        decoy_residue_energies=[
            {f"A{k + 1}": 0.5 + 0.1 * k for k in range(6)},
            {f"A{k + 1}": 0.7 + 0.1 * k for k in range(6)},
        ],
    )
    ap.write_tertiary_frustration(
        str(tmp_path / "tertiary_frustration.dat"), geom, engine, seq_sep=0
    )
    _write_equivalences(tmp_path, geom)
    renum_files("native", str(tmp_path), "configurational")
    df = pd.read_csv(tmp_path / "native.pdb_configurational", sep=r"\s+")
    assert list(df.columns) == CONTACT_COLS
    assert set(df["FrstState"]).issubset({"minimally", "neutral", "highly"})


def test_mutational_routes_to_14col_contact_table(tmp_path):
    """Mutational (EXPERIMENTAL): per-contact decoy statistics -> shared contact
    writer (via per-contact summaries) -> shared process_results -> 14-column table.
    Demonstrates the per-contact statistic reuses the SAME writer as configurational,
    not a forked one."""
    import pandas as pd
    from frustrapy.utils.helpers import renum_files
    from frustrapy.backends import atomic_engine as ae

    geom = _toy_geometry()
    contacts = ap.select_contacts(geom, seq_sep=0)

    # Mock the per-contact decoy ensemble: generate specs, score them with a stub,
    # pool per contact, build one ContactEnergySummary per contact (in writer order).
    strat = am.get_decoy_strategy("mutational")
    groups = strat.generate(
        "".join(geom.aa), contacts=contacts, n_decoys_per_contact=6, seed=1
    )

    def score_fn(spec, group):
        i, j = group.key
        return float(ord(spec.sequence[i]) + ord(spec.sequence[j])) / 50.0

    stats = am.aggregate_groups(groups, score_fn)
    summaries = []
    for (i, j) in contacts:
        mean, std, _ = stats[(i, j)]
        # Native energy is a stub (maintainer Rosetta value); shape is what matters.
        native = -2.0
        summaries.append(
            ae.ContactEnergySummary(
                i_key=geom.cid_list[i], j_key=geom.cid_list[j],
                native_energy=native, decoy_mean=mean, decoy_std=std, n_decoys=6,
            )
        )

    ap.write_tertiary_frustration(
        str(tmp_path / "tertiary_frustration.dat"), geom, seq_sep=0, summaries=summaries
    )
    _write_equivalences(tmp_path, geom)
    renum_files("native", str(tmp_path), "mutational")
    df = pd.read_csv(tmp_path / "native.pdb_mutational", sep=r"\s+")
    assert list(df.columns) == CONTACT_COLS
    assert len(df) == len(contacts)


def test_singleresidue_routes_to_8col_table(tmp_path):
    """Single-residue (EXPERIMENTAL): per-site decoy statistics -> shared
    single-residue writer -> shared process_results -> canonical 8-column
    single-residue table (no FrstState column)."""
    import pandas as pd
    from frustrapy.utils.helpers import renum_files

    geom = _toy_geometry()
    sites = list(range(geom.n_residues))

    strat = am.get_decoy_strategy("singleresidue")
    groups = strat.generate("".join(geom.aa), sites=sites)

    def score_fn(spec, group):
        i = group.key
        return float(ord(spec.sequence[i])) / 20.0

    stats = am.aggregate_groups(groups, score_fn)
    site_summaries = []
    for i in sites:
        mean, std, _ = stats[i]
        site_summaries.append(
            ap.SiteEnergySummary(site_index=i, native_energy=-1.5, decoy_mean=mean, decoy_std=std)
        )

    written = ap.write_singleresidue_dat(
        str(tmp_path / "tertiary_frustration.dat"), geom, site_summaries
    )
    assert written == sites
    _write_equivalences(tmp_path, geom)
    renum_files("native", str(tmp_path), "singleresidue")
    df = pd.read_csv(tmp_path / "native.pdb_singleresidue", sep=r"\s+")
    assert list(df.columns) == SINGLERES_COLS
    assert "FrstState" not in df.columns  # single-residue table carries no class column
    assert len(df) == len(sites)


def test_singleresidue_dat_uses_shared_sign_flip(tmp_path):
    """The single-residue writer computes FrstIndex with the SAME sign-flipped
    Z-score (atomic_frustration_index) the contact writer uses: a favorable (very
    negative) native energy -> positive index."""
    geom = _toy_geometry()
    summ = ap.SiteEnergySummary(site_index=0, native_energy=-9.0, decoy_mean=-5.0, decoy_std=2.0)
    ap.write_singleresidue_dat(str(tmp_path / "tertiary_frustration.dat"), geom, [summ])
    line = [
        l for l in (tmp_path / "tertiary_frustration.dat").read_text().splitlines()
        if not l.startswith("#")
    ][0]
    f_i = float(line.split()[10])  # singleresidue layout: f_i is the last column
    assert f_i == pytest.approx(ap.atomic_frustration_index(-9.0, -5.0, 2.0))
    assert f_i > 0  # favorable native -> minimally frustrated (positive)


# ---------------------------------------------------------------------------
# Cutoffs are preserved and never collapsed (the mission's explicit guard).
# ---------------------------------------------------------------------------

def test_cutoffs_unchanged_and_not_collapsed():
    from frustrapy.core import constants as C

    # Contact cutoffs (configurational/mutational) and the single-residue plot cutoff
    # are distinct and unchanged by the atomic modes.
    assert C.FRST_HIGHLY_MAX == -1.0
    assert C.FRST_MINIMALLY_MIN_CONTACT == 0.78
    assert C.FRST_MINIMALLY_MIN_SINGLERES == 0.58
    assert C.FRST_MINIMALLY_MIN_CONTACT != C.FRST_MINIMALLY_MIN_SINGLERES
