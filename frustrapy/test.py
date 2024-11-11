# @title Frustratometer in Python
mode = "singleresidue"  # @param ["configurational", "singleresidue", "mutational"]
pdbs_dir = "/content"  # @param {type:"string"}
results_dir = (
    "/home/ceramirez/github/frustrapy/Results_example"  # @param {type:"string"}
)
example = True  # @param {type:"boolean"}
overwrite = False  # @param {type:"boolean"}
debug = True  # @param {type:"boolean"}

import sys

# /home/ceramirez/github/frustrapy/frustrapy/functions.py
# Import the functions from the frustrapy package
import functions as frustrapy_functions

# If the example is True, we will download the example files
import subprocess

if example:
    # subprocess.run(
    #    ["wget", "-q", "http://www.rcsb.org/pdb/files/1fhj.pdb", "-O", "1fhj.pdb"]
    # )
    # subprocess.run(
    #    ["wget", "-q", "http://www.rcsb.org/pdb/files/2dn1.pdb", "-O", "2dn1.pdb"]
    # )
    # subprocess.run(
    #    ["wget", "-q", "http://www.rcsb.org/pdb/files/1m6k.pdb", "-O", "1m6k.pdb"]
    # )

    pdbs_dir = "/home/ceramirez/github/frustrapy"
    results_dir = "/home/ceramirez/github/frustrapy/Results_example"
    # Remove any previous results
    subprocess.run(["rm", "-rf", "/home/ceramirez/github/frustrapy/Results_example/*"])

if overwrite:
    if example:
        subprocess.run(["rm", "-rf", "/home/ceramirez/github/frustrapy/Results/*"])
    else:
        import os

        # Convert the results_dir to an absolute path
        results_dir = os.path.abspath(results_dir)
        subprocess.run(["rm", "-rf", f"{results_dir}/*"])

plots_dir_dict = frustrapy_functions.dir_frustration(
    pdbs_dir=pdbs_dir, mode=mode, results_dir=results_dir, debug=debug
)
