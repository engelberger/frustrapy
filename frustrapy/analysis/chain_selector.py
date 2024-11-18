from Bio.PDB import Select
from typing import List


class NonHetSelect(Select):
    """Selector for non-HET atoms."""

    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0


class ChainSelect(Select):
    """Selector for specific chains."""

    def __init__(self, selected_chains: List[str]):
        self.selected_chains = selected_chains

    def accept_chain(self, chain_obj):
        return chain_obj.id in self.selected_chains
