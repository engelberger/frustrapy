"""``frustrapy`` command-line tool.

Thin Typer wrappers over the :mod:`frustrapy.sdk` facade — every subcommand parses
arguments, calls one SDK function, and renders the result with Rich. No analysis
logic lives here.

Subcommands
-----------
* ``single``  — one PDB through one frustration mode (``calculate_frustration``).
* ``batch``   — every PDB in a directory (``dir_frustration``).
* ``evo``     — evolutionary frustration for a protein family (``analyze_family``).
* ``mutate``  — saturation-mutagenesis scan of one residue (``mutate_res_parallel``).

``typer`` is imported lazily so that merely importing this module on a bare install
fails gracefully with an actionable message instead of an opaque ``ImportError``.
The core library never imports this module, so ``import frustrapy`` does not require
``typer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# The CLI extra (`pip install "frustrapy[cli]"`) provides typer; rich is already a
# core dependency. Guard the import so the console script can print a helpful hint
# rather than a raw traceback when the extra is missing.
try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _CLI_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover - exercised only on a bare install
    typer = None  # type: ignore[assignment]
    _CLI_IMPORT_ERROR = exc


# Frustration classes carried in the FrstState column of contact tables, in the
# order we want them displayed.
_FRST_STATES = ("minimally", "neutral", "highly")


if typer is not None:
    console = Console()
    err_console = Console(stderr=True)

    app = typer.Typer(
        name="frustrapy",
        help=(
            "Compute local energetic frustration in protein structures "
            "(a Python reimplementation of frustratometeR)."
        ),
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="rich",
    )

    def _version_callback(value: bool) -> None:
        if value:
            from frustrapy import __version__

            console.print(f"frustrapy {__version__}")
            raise typer.Exit()

    @app.callback()
    def _main(
        version: bool = typer.Option(
            False,
            "--version",
            "-V",
            help="Show the installed FrustraPy version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ) -> None:
        """FrustraPy — local energetic frustration analysis."""

    # ----------------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------------- #
    def _fail(message: str, code: int = 1) -> "typer.Exit":
        """Render an error panel and return an Exit carrying ``code``."""
        err_console.print(
            Panel(message, title="Error", border_style="red", expand=False)
        )
        return typer.Exit(code)

    def _summary_table(df, mode: str) -> Table:
        """Build a Rich table summarising one frustration result."""
        table = Table(title=f"Frustration summary ({mode})", expand=False)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        n_rows = len(df)
        unit = "residues" if mode == "singleresidue" else "contacts"
        table.add_row(f"Scored {unit}", str(n_rows))
        if "FrstState" in df.columns and n_rows:
            counts = df["FrstState"].value_counts()
            for state in _FRST_STATES:
                n = int(counts.get(state, 0))
                pct = 100.0 * n / n_rows if n_rows else 0.0
                table.add_row(f"{state} frustrated", f"{n}  ({pct:.1f}%)")
        return table

    # ----------------------------------------------------------------------- #
    # single
    # ----------------------------------------------------------------------- #
    @app.command()
    def single(
        pdb_file: Path = typer.Argument(
            ..., exists=True, dir_okay=False, readable=True, help="Input PDB file."
        ),
        mode: str = typer.Option(
            "configurational",
            "--mode",
            "-m",
            help="Frustration index: configurational, mutational, or singleresidue.",
        ),
        seq_dist: int = typer.Option(
            12, "--seq-dist", help="Sequence separation for contacts (3 or 12)."
        ),
        chain: Optional[str] = typer.Option(
            None, "--chain", "-c", help="Restrict analysis to this chain."
        ),
        results_dir: Optional[Path] = typer.Option(
            None, "--results-dir", "-o", help="Output directory (default: cwd)."
        ),
        graphics: bool = typer.Option(
            False, "--graphics/--no-graphics", help="Generate Plotly graphics."
        ),
        n_cpus: Optional[int] = typer.Option(
            None, "--n-cpus", help="CPU cores for the mutation pool (singleresidue)."
        ),
    ) -> None:
        """Run one PDB through one frustration mode."""
        import frustrapy.sdk as fp

        if mode not in ("configurational", "mutational", "singleresidue"):
            raise _fail(
                f"Unknown mode '{mode}'. Choose configurational, mutational, "
                "or singleresidue."
            )
        if seq_dist not in (3, 12):
            raise _fail(f"--seq-dist must be 3 or 12, got {seq_dist}.")

        with console.status(
            f"[bold]Computing {mode} frustration for {pdb_file.name}…", spinner="dots"
        ):
            try:
                pdb, _plots, _density, _single = fp.calculate_frustration(
                    pdb_file=str(pdb_file),
                    mode=mode,
                    seq_dist=seq_dist,
                    chain=chain,
                    results_dir=str(results_dir) if results_dir else None,
                    graphics=graphics,
                    visualization=graphics,
                    debug="ERROR",
                )
            except Exception as exc:  # surface the failure as a clean message
                raise _fail(f"Frustration calculation failed: {exc}")

        try:
            df = fp.get_frustration(pdb)
            console.print(_summary_table(df, mode))
        except Exception:  # the run succeeded even if the summary read did not
            pass
        console.print(
            Panel(
                f"Output written to [bold]{pdb.job_dir}[/bold]\n"
                f"Tables under FrustrationData/",
                title="Done",
                border_style="green",
                expand=False,
            )
        )

    # ----------------------------------------------------------------------- #
    # batch
    # ----------------------------------------------------------------------- #
    @app.command()
    def batch(
        pdbs_dir: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory of PDB files.",
        ),
        mode: str = typer.Option(
            "configurational", "--mode", "-m", help="Frustration index."
        ),
        seq_dist: int = typer.Option(
            12, "--seq-dist", help="Sequence separation for contacts (3 or 12)."
        ),
        results_dir: Optional[Path] = typer.Option(
            None, "--results-dir", "-o", help="Output directory (default: cwd)."
        ),
        n_procs: Optional[int] = typer.Option(
            None, "--n-procs", help="Structures to process concurrently."
        ),
        n_cpus: Optional[int] = typer.Option(
            None, "--n-cpus", help="Inner CPU budget per structure."
        ),
        graphics: bool = typer.Option(
            False, "--graphics/--no-graphics", help="Generate Plotly graphics."
        ),
    ) -> None:
        """Run every PDB in a directory."""
        import frustrapy.sdk as fp

        if mode not in ("configurational", "mutational", "singleresidue"):
            raise _fail(f"Unknown mode '{mode}'.")
        if seq_dist not in (3, 12):
            raise _fail(f"--seq-dist must be 3 or 12, got {seq_dist}.")

        pdbs = sorted(p.name for p in pdbs_dir.glob("*.pdb"))
        if not pdbs:
            raise _fail(f"No .pdb files found in {pdbs_dir}.")

        with console.status(
            f"[bold]Computing {mode} frustration for {len(pdbs)} structures…",
            spinner="dots",
        ):
            try:
                fp.dir_frustration(
                    pdbs_dir=str(pdbs_dir),
                    mode=mode,
                    seq_dist=seq_dist,
                    results_dir=str(results_dir) if results_dir else None,
                    graphics=graphics,
                    visualization=graphics,
                    n_procs=n_procs,
                    n_cpus=n_cpus,
                    debug=False,
                )
            except Exception as exc:
                raise _fail(f"Batch frustration failed: {exc}")

        out = results_dir if results_dir else Path.cwd()
        console.print(
            Panel(
                f"Processed [bold]{len(pdbs)}[/bold] structures.\n"
                f"Output under [bold]{out}[/bold]",
                title="Done",
                border_style="green",
                expand=False,
            )
        )

    # ----------------------------------------------------------------------- #
    # evo
    # ----------------------------------------------------------------------- #
    @app.command()
    def evo(
        fasta_file: Path = typer.Argument(
            ..., exists=True, dir_okay=False, readable=True, help="MSA in FASTA format."
        ),
        job_id: str = typer.Option(..., "--job-id", "-j", help="Analysis identifier."),
        pdb_dir: Path = typer.Option(
            ...,
            "--pdb-dir",
            "-p",
            exists=True,
            file_okay=False,
            help="Directory of per-member PDB files.",
        ),
        reference_pdb: Optional[str] = typer.Option(
            None, "--reference-pdb", "-r", help="Reference member identifier."
        ),
        results_dir: Optional[Path] = typer.Option(
            None, "--results-dir", "-o", help="Output directory."
        ),
        contact_maps: bool = typer.Option(
            False, "--contact-maps", help="Generate contact-map plots."
        ),
        n_procs: Optional[int] = typer.Option(
            None, "--n-procs", help="Per-structure calculations to run concurrently."
        ),
    ) -> None:
        """Evolutionary frustration for a protein family (FrustraEvo)."""
        import frustrapy.sdk as fp

        with console.status(
            f"[bold]Analyzing family '{job_id}'…", spinner="dots"
        ):
            try:
                result = fp.analyze_family(
                    fasta_file=str(fasta_file),
                    job_id=job_id,
                    reference_pdb=reference_pdb,
                    pdb_dir=str(pdb_dir),
                    contact_maps=contact_maps,
                    results_dir=str(results_dir) if results_dir else None,
                    n_procs=n_procs,
                )
            except Exception as exc:
                raise _fail(f"Family analysis failed: {exc}")

        table = Table(title=f"FrustraEvo summary ({job_id})", expand=False)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        summary = {}
        if isinstance(result, dict):
            summary = result.get("contacts", {}).get("summary", {})
        for label, key in (
            ("total contacts", "total_contacts"),
            ("minimally frustrated", "minimally_frustrated"),
            ("neutrally frustrated", "neutrally_frustrated"),
            ("maximally frustrated", "maximally_frustrated"),
        ):
            if key in summary:
                table.add_row(label, str(summary[key]))
        out = result.get("output_dir") if isinstance(result, dict) else None
        console.print(table)
        console.print(
            Panel(
                f"Output written to [bold]{out}[/bold]",
                title="Done",
                border_style="green",
                expand=False,
            )
        )

    # ----------------------------------------------------------------------- #
    # mutate
    # ----------------------------------------------------------------------- #
    @app.command()
    def mutate(
        pdb_file: Path = typer.Argument(
            ..., exists=True, dir_okay=False, readable=True, help="Input PDB file."
        ),
        res: int = typer.Option(..., "--res", help="Residue number to scan."),
        chain: str = typer.Option("A", "--chain", "-c", help="Chain of the residue."),
        method: str = typer.Option(
            "threading",
            "--method",
            help="Mutation backend: threading, modeller, or pyrosetta.",
        ),
        results_dir: Optional[Path] = typer.Option(
            None, "--results-dir", "-o", help="Output directory."
        ),
        n_cpus: Optional[int] = typer.Option(
            None, "--n-cpus", help="CPU cores for the mutation pool."
        ),
    ) -> None:
        """Saturation-mutagenesis scan of one residue (all 20 amino acids)."""
        import frustrapy.sdk as fp

        if method not in ("threading", "modeller", "pyrosetta"):
            raise _fail(
                f"Unknown method '{method}'. Choose threading, modeller, "
                "or pyrosetta."
            )

        with console.status(
            f"[bold]Scanning residue {res}{chain} ({method})…", spinner="dots"
        ):
            try:
                pdb, _plots, _density, _single = fp.calculate_frustration(
                    pdb_file=str(pdb_file),
                    mode="singleresidue",
                    residues={chain: [res]},
                    results_dir=str(results_dir) if results_dir else None,
                    graphics=False,
                    visualization=False,
                    debug="ERROR",
                )
                fp.mutate_res_parallel(
                    pdb, res_num=res, chain=chain, method=method, n_cpus=n_cpus
                )
            except Exception as exc:
                raise _fail(f"Mutation scan failed: {exc}")

        out = Path(pdb.job_dir) / "MutationsData"
        console.print(
            Panel(
                f"Scanned residue [bold]{res}{chain}[/bold] over 20 amino acids "
                f"({method}).\nOutput under [bold]{out}[/bold]",
                title="Done",
                border_style="green",
                expand=False,
            )
        )


def main() -> None:
    """Console-script entry point (``frustrapy``)."""
    if typer is None:
        import sys

        sys.stderr.write(
            "The frustrapy CLI requires the 'cli' extra.\n"
            "Install it with:  pip install \"frustrapy[cli]\"\n"
            f"(import error: {_CLI_IMPORT_ERROR})\n"
        )
        raise SystemExit(1)
    app()


if __name__ == "__main__":
    main()
