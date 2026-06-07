"""Command-line interface for FrustraPy.

The CLI is an optional component: it depends on ``typer`` (and ``rich``, already a
core dependency) which ship in the ``cli`` extra::

    pip install "frustrapy[cli]"

The console-script entry point is :func:`frustrapy.cli.main.main`. Importing this
subpackage does not import ``typer``; the core library never requires it. The
top-level ``frustrapy`` package does not import this module, so ``import frustrapy``
stays free of any CLI dependency.
"""
