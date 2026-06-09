"""Molecular-visualization parity and cross-surface consistency.

Three surfaces must agree for every mode (configurational / mutational /
single-residue): the PyMOL ``.pml`` script, the ChimeraX ``.cxc`` (+ ``.pb``)
script, and the class-colored Plotly figure. They all derive their classes and
colors from the same parsed FrustrationData table at the audited cutoffs
(contacts -1 / 0.78, single-residue plot -1 / 0.58).

Fast tests build synthetic tables and need no LAMMPS. The PyMOL parity tests
compare the pure-Python generator against the legacy
``GenerateVisualizations.pl`` oracle and are skipped if ``perl`` is absent. The
slow (``e2e``) tests run the real engine on 1crn -- always on a private temp
copy of the fixture, never the committed file (single-residue mode rewrites its
input PDB in place; see the project conventions).
"""

import os
import shutil
import subprocess

import pandas as pd
import pytest

from frustrapy.visualization.pymol_script import (
    generate_contact_pml,
    generate_singleresidue_pml,
    write_pml,
)
from frustrapy.visualization.chimerax_script import write_cxc, generate_pb
from frustrapy.visualization.frustration_data import (
    contact_links,
    contact_links_from_df,
    residue_colors_from_df,
    class_counts_contacts,
    class_counts_residues,
)
from frustrapy.visualization.plots import plot_frustration_classes, figure_class_counts

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CRN_PDB = os.path.join(DATA_DIR, "1crn.pdb")
PERL_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "frustrapy",
    "core",
    "scripts",
    "GenerateVisualizations.pl",
)
HAS_PERL = shutil.which("perl") is not None

CONTACT_COLUMNS = [
    "Res1", "Res2", "ChainRes1", "ChainRes2", "Welltype", "FrstState",
]


def _synthetic_contact_df():
    """A two-chain contact table exercising every PyMOL draw branch."""
    rows = [
        (1, 3, "A", "A", "short", "highly"),            # red draw_links
        (2, 10, "A", "B", "long", "minimally"),         # green draw_links, cross-chain
        (4, 6, "A", "A", "water-mediated", "minimally"),  # min distance
        (43, 45, "B", "B", "water-mediated", "highly"),   # max distance
        (5, 7, "A", "A", "long", "neutral"),            # excluded (neutral)
    ]
    return pd.DataFrame(rows, columns=CONTACT_COLUMNS)


def _synthetic_singleresidue_df():
    rows = [
        (1, "A", "T", -1.5),   # highly  (<= -1)
        (2, "A", "C", 0.2),    # neutral (-1 < x < 0.58)
        (3, "A", "P", 0.9),    # minimally (>= 0.58)
        (4, "A", "G", 0.6),    # minimally (>= 0.58, below the 0.78 contact cutoff)
        (5, "A", "S", -0.5),   # neutral
    ]
    return pd.DataFrame(rows, columns=["Res", "ChainRes", "AA", "FrstIndex"])


def _aux_from_contact_df(df, path):
    """Write the legacy ``*_auxiliar`` file (red/green rows only) for the Perl."""
    with open(path, "w") as fh:
        for _, r in df.iterrows():
            state = r["FrstState"]
            if state == "highly":
                color = "red"
            elif state == "minimally":
                color = "green"
            else:
                continue
            fh.write(
                f'{r["Res1"]} {r["Res2"]} {r["ChainRes1"]} {r["ChainRes2"]} '
                f'{r["Welltype"]} {color}\n'
            )


def _run_perl_pml(aux_path, pdb_base, out_dir, mode):
    subprocess.run(
        ["perl", PERL_SCRIPT, os.path.basename(aux_path), pdb_base, out_dir, mode],
        check=True,
    )
    return os.path.join(out_dir, f"{pdb_base}.pdb_{mode}.pml")


# --------------------------------------------------------------------------- #
# Fast unit tests (no LAMMPS)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_PERL, reason="perl not installed")
def test_pml_synthetic_parity_vs_perl(tmp_path):
    """The Python contact ``.pml`` is byte-identical to GenerateVisualizations.pl
    on a synthetic multi-chain table that hits every draw branch (direct
    red/green links, cross-chain, and water-mediated min/max distances)."""
    df = _synthetic_contact_df()
    py_pml = generate_contact_pml(contact_links_from_df(df), "test")

    aux = os.path.join(tmp_path, "test_configurational.pdb_auxiliar")
    _aux_from_contact_df(df, aux)
    perl_pml = open(
        _run_perl_pml(aux, "test", str(tmp_path), "configurational")
    ).read()

    assert py_pml == perl_pml


def test_contact_links_filter_and_color():
    """Only highly/minimally contacts are drawn; colors follow the contact map."""
    links = contact_links_from_df(_synthetic_contact_df())
    assert len(links) == 4  # the neutral row is excluded
    by_color = {(l.res1, l.res2): l.color for l in links}
    assert by_color[("1", "3")] == "red"
    assert by_color[("2", "10")] == "green"


def test_singleresidue_pml_uses_058_cutoff():
    """Single-residue coloring uses the -1 / 0.58 plot cutoffs, not 0.78:
    a residue at FrstIndex 0.6 is green (minimally), which the 0.78 contact
    cutoff would have called neutral."""
    residues = residue_colors_from_df(_synthetic_singleresidue_df())
    state = {r.res: r.state for r in residues}
    assert state["1"] == "highly"
    assert state["2"] == "neutral"
    assert state["3"] == "minimally"
    assert state["4"] == "minimally"  # 0.6 >= 0.58
    pml = generate_singleresidue_pml(residues, "test")
    assert "color red, test and chain A and resi 1" in pml
    assert "color green, test and chain A and resi 4" in pml


def test_pb_contact_counts_match_table():
    """The ChimeraX pseudobond file colors one bond per red/green contact."""
    links = contact_links_from_df(_synthetic_contact_df())
    pb = generate_pb(links)
    body = [l for l in pb.splitlines() if not l.startswith(";")]
    red = sum(1 for l in body if l.endswith("red"))
    green = sum(1 for l in body if l.endswith("green"))
    counts = class_counts_contacts(links)
    assert red == counts["highly"]
    assert green == counts["minimally"]


def test_cross_surface_counts_consistent_synthetic(tmp_path):
    """PyMOL, ChimeraX and the parsed table agree on per-class contact counts."""
    df = _synthetic_contact_df()
    table = os.path.join(tmp_path, "test.pdb_configurational")
    # write a table the generators can read back (whitespace separated)
    df.to_csv(table, sep=" ", index=False)

    write_pml(table, os.path.join(tmp_path, "out.pml"), "test", "configurational")
    write_cxc(table, os.path.join(tmp_path, "out.cxc"), "test", "configurational")

    pml = open(os.path.join(tmp_path, "out.pml")).read()
    pml_green = pml.count("color=green")  # draw_links lines
    pml_green += sum(
        1 for l in pml.splitlines() if l.startswith("distance min_frst_wm")
    )
    pml_red = pml.count("color=red")
    pml_red += sum(
        1 for l in pml.splitlines() if l.startswith("distance max_frst_wm")
    )

    pb = open(os.path.join(tmp_path, "out.pb")).read()
    body = [l for l in pb.splitlines() if not l.startswith(";")]
    pb_red = sum(1 for l in body if l.endswith("red"))
    pb_green = sum(1 for l in body if l.endswith("green"))

    counts = class_counts_contacts(contact_links(table))
    assert pml_red == pb_red == counts["highly"]
    assert pml_green == pb_green == counts["minimally"]


# --------------------------------------------------------------------------- #
# Slow end-to-end tests (real LAMMPS engine on a private 1crn copy)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def viz_runs(tmp_path_factory):
    """Run 1crn in all three modes once, on a private copy of the fixture.

    Returns ``{mode: job_dir}``. Uses ``debug='ERROR'`` so temp files (incl. the
    legacy ``*_auxiliar``) survive for the Perl-parity comparison. Never touches
    the committed fixture (single-residue mode rewrites its input in place).
    """
    import warnings
    import frustrapy

    root = tmp_path_factory.mktemp("viz")
    pdb_copy = os.path.join(root, "1crn.pdb")
    shutil.copy2(CRN_PDB, pdb_copy)

    runs = {}
    for mode in ["configurational", "mutational", "singleresidue"]:
        results_dir = os.path.join(root, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frustrapy.calculate_frustration(
                pdb_file=pdb_copy,
                mode=mode,
                results_dir=results_dir,
                graphics=False,
                visualization=True,
                debug="ERROR",
            )
        runs[mode] = os.path.join(results_dir, "1crn.done")
    return runs


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["configurational", "mutational"])
@pytest.mark.skipif(not HAS_PERL, reason="perl not installed")
def test_1crn_pml_byte_identical_to_perl(viz_runs, tmp_path, mode):
    """End-to-end: the generated PyMOL ``.pml`` is byte-identical to the Perl
    oracle run on the same ``*_auxiliar`` file produced by the run."""
    job_dir = viz_runs[mode]
    py_pml = os.path.join(
        job_dir, "VisualizationScripts", f"1crn.pdb_{mode}.pml"
    )
    aux = os.path.join(job_dir, f"1crn_{mode}.pdb_auxiliar")
    assert os.path.exists(py_pml), py_pml
    assert os.path.exists(aux), aux

    shutil.copy2(aux, os.path.join(tmp_path, os.path.basename(aux)))
    perl_pml = _run_perl_pml(
        os.path.join(tmp_path, os.path.basename(aux)), "1crn", str(tmp_path), mode
    )
    assert open(py_pml).read() == open(perl_pml).read()


@pytest.mark.slow
@pytest.mark.parametrize(
    "mode", ["configurational", "mutational", "singleresidue"]
)
def test_1crn_cross_surface_consistency(viz_runs, mode):
    """For each mode the three surfaces exist, are non-empty, and agree on the
    per-class contact/residue counts derived from the parsed table."""
    from types import SimpleNamespace

    job_dir = viz_runs[mode]
    vis_dir = os.path.join(job_dir, "VisualizationScripts")
    table = os.path.join(job_dir, "FrustrationData", f"1crn.pdb_{mode}")

    pml_path = os.path.join(vis_dir, f"1crn.pdb_{mode}.pml")
    cxc_path = os.path.join(vis_dir, f"1crn.pdb_{mode}.cxc")
    for p in (pml_path, cxc_path):
        assert os.path.exists(p), p
        assert os.path.getsize(p) > 0

    # Plotly surface
    pdb = SimpleNamespace(job_dir=job_dir, pdb_base="1crn", mode=mode)
    fig = plot_frustration_classes(pdb)
    fig_counts = figure_class_counts(fig)

    df = pd.read_csv(table, sep=r"\s+")

    if mode == "singleresidue":
        residues = residue_colors_from_df(df)
        counts = class_counts_residues(residues)
        # every residue is colored in the .pml; counts agree with the figure
        assert sum(counts.values()) == len(df)
        for cls in ("highly", "neutral", "minimally"):
            assert fig_counts[cls] == counts[cls]
        # single-residue: no .pb file (per-residue coloring, not contacts)
        assert not os.path.exists(os.path.join(vis_dir, f"1crn.pdb_{mode}.pb"))
    else:
        counts = class_counts_contacts(contact_links_from_df(df))
        pb_path = os.path.join(vis_dir, f"1crn.pdb_{mode}.pb")
        assert os.path.exists(pb_path)
        body = [l for l in open(pb_path) if not l.startswith(";")]
        pb_red = sum(1 for l in body if l.strip().endswith("red"))
        pb_green = sum(1 for l in body if l.strip().endswith("green"))

        pml = open(pml_path).read()
        pml_red = pml.count("color=red") + sum(
            1 for l in pml.splitlines() if l.startswith("distance max_frst_wm")
        )
        pml_green = pml.count("color=green") + sum(
            1 for l in pml.splitlines() if l.startswith("distance min_frst_wm")
        )

        assert pml_red == pb_red == fig_counts["highly"] == counts["highly"]
        assert pml_green == pb_green == fig_counts["minimally"] == counts["minimally"]
