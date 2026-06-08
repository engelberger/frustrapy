"""Python 3 port of the atomic (Rosetta) Frustratometer post-processor.

This is a faithful, minimal port of
``/workspace/atomic_frustratometer_ref/example_input/Frust_Post_public.py``
(copyright Mingchen Chen, 2019; Academic Free License v3.0). It reads the shipped
native + decoy Rosetta ResResE logs and the input structure and writes
``tertiary_frustration.dat`` in the reference's own column layout.

Only mechanical Python 2 -> 3 changes were made (print -> print(), np.float ->
float, tab/space normalisation) plus removal of the matplotlib / scipy plotting
imports and the per-residue debug prints; the numeric path (energy extraction,
decoy statistics, Z-score, .dat layout) is unchanged. The VMD .tcl and PyMOL .pml
visualisation outputs are still written but are not part of the parity fixture.

Run from a directory that contains ``3gso.pdb`` (a copy of the scored structure),
``native.log`` and ``1.log`` .. ``N.log``:

    python3 Frust_Post_public_py3.py 100 -2.5 0.5 50 9 0 Function1 -1.5 0.5
"""

import sys
import numpy as np
from math import sqrt

import Bio.PDB
from Bio.PDB.PDBParser import PDBParser


def vector(p1, p2):
    return [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]


def vabs(a):
    return sqrt(pow(a[0], 2) + pow(a[1], 2) + pow(a[2], 2))


def get_Atom_Ligand(residue):
    temp_sum_update = 99999999
    atom_update = None
    for atom1 in residue:
        temp_sum = 0.0
        for atom2 in residue:
            diff_vector = atom1.coord - atom2.coord
            temp_sum = np.sum(diff_vector * diff_vector) + temp_sum
        if temp_sum < temp_sum_update:
            temp_sum_update = temp_sum
            atom_update = atom1
    return atom_update


def calc_residue_dist_new(residue_one, residue_two):
    dist = 999999
    temp = dist
    for atom1 in residue_one:
        for atom2 in residue_two:
            diff_vector = atom1.coord - atom2.coord
            temp = np.sqrt(np.sum(diff_vector * diff_vector))
            if temp < dist:
                dist = temp
    return temp


def calc_dist_matrix(ca_atoms):
    reslen = len(ca_atoms)
    answer = []
    se_map = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
              "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
              "TYR", "VAL", "MSE"]
    p = PDBParser(PERMISSIVE=1)
    pdbcode = '3gso'
    s = p.get_structure(pdbcode, pdbcode + '.pdb')
    chains = s[0].get_list()
    for chain1 in chains:
        for res1 in chain1:
            for chain2 in chains:
                for res2 in chain2:
                    if res1.has_id('CB') == 1 and res2.has_id('CB') == 1 and (res1.get_resname() in se_map) and (res2.get_resname() in se_map):
                        answer.append(vabs(vector(res1['CB'].get_coord(), res2['CB'].get_coord())))
                    if res1.has_id('CB') == 1 and res2.has_id('CB') == 0 and res2.has_id('CA') == 1 and (res1.get_resname() in se_map) and (res2.get_resname() in se_map):
                        answer.append(vabs(vector(res1['CB'].get_coord(), res2['CA'].get_coord())))
                    if res1.has_id('CB') == 0 and res1.has_id('CA') == 1 and res2.has_id('CB') == 1 and (res1.get_resname() in se_map) and (res2.get_resname() in se_map):
                        answer.append(vabs(vector(res1['CA'].get_coord(), res2['CB'].get_coord())))
                    if res1.has_id('CB') == 0 and res1.has_id('CA') == 1 and res2.has_id('CB') == 0 and res2.has_id('CA') == 1 and (res1.get_resname() in se_map) and (res2.get_resname() in se_map):
                        answer.append(vabs(vector(res1['CA'].get_coord(), res2['CA'].get_coord())))
                    if (res1.get_resname() not in se_map) or (res2.get_resname() not in se_map):
                        result_temp = calc_residue_dist_new(res1, res2)
                        answer.append(result_temp)
    answer_new = np.array(answer).reshape(reslen, reslen)
    return answer_new


def get_index(pdbcode):
    se_map = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
              "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
              "TYR", "VAL", "MSE"]
    se_map_b = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K",
                "M", "F", "P", "S", "T", "W", "Y", "V", "M"]
    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure(pdbcode, pdbcode + '.pdb')
    chains = s[0].get_list()
    ca_atoms = []
    cid_list = []
    atom_list = []
    lig_pos = []
    res_list_name = []

    for chain in chains:
        for res in chain:
            cid_list.append(chain.id + str(res.get_id()[1]))
            if (res.get_resname() in se_map) and (res.get_resname() == 'GLY' or res.has_id('CB') == 0) and res.has_id('CA') == 1:
                ca_atoms.append(res['CA'].get_coord())
                atom_list.append('CA')
                lig_pos.append(0)
                res_list_name.append(se_map_b[se_map.index(res.get_resname())])
            if (res.get_resname() in se_map) and res.has_id('CB'):
                ca_atoms.append(res['CB'].get_coord())
                atom_list.append('CA')
                lig_pos.append(0)
                res_list_name.append(se_map_b[se_map.index(res.get_resname())])
            if (res.get_resname() in se_map) and res.has_id('CB') == 0 and res.has_id('CA') == 0:
                ca_atoms.append(res['O'].get_coord())
                atom_list.append('O')
                lig_pos.append(0)
                res_list_name.append(se_map_b[se_map.index(res.get_resname())])
            if res.get_resname() not in se_map:
                atom1 = get_Atom_Ligand(res)
                ca_atoms.append(res[atom1.get_name()].get_coord())
                atom_list.append(atom1.get_name())
                lig_pos.append(1)
                res_list_name.append("X")
    dist_matrix = calc_dist_matrix(ca_atoms)
    return dist_matrix, cid_list, atom_list, lig_pos, res_list_name, ca_atoms


def _ene_from_strs(strs, scheme):
    if scheme == "Function1":  # leave out the rep term only
        return float(strs[-1]) - 1.0 * (float(strs[4])) - float(strs[10]) - float(strs[15])
    if scheme == "Function2":  # leave both rep and atr terms
        return float(strs[-1]) - 1.0 * (float(strs[4])) - 1.0 * (float(strs[3])) - float(strs[10]) - float(strs[15])
    if scheme == "Packing":
        return float(strs[-1]) - float(strs[10]) - float(strs[15])
    raise ValueError("unknown scheme " + str(scheme))


def read_log(fname, cid_list, scheme):
    fin = open(fname, 'r')
    reslen = len(cid_list)
    mat = np.zeros((reslen, reslen))
    ene_res = np.zeros((reslen,))
    for line in fin:
        strs = line.split()
        if strs[1] != 'Res1' and strs[1] != 'nonzero' and float(strs[4]) <= 5.000:
            ind1 = cid_list.index(strs[1][2:])
            ind2 = cid_list.index(strs[2][2:])
            ene = _ene_from_strs(strs, scheme)
            ene_res[ind1] += 0.5 * ene
            ene_res[ind2] += 0.5 * ene
    for i in range(reslen):
        for j in range(reslen):
            mat[i][j] = ene_res[i] + ene_res[j]
            mat[j][i] = ene_res[i] + ene_res[j]
    return mat


def read_nat_log(fname, cid_list, scheme):
    fin = open(fname, 'r')
    reslen = len(cid_list)
    mat = 999 * np.ones((reslen, reslen))
    ene_res = np.zeros((reslen,))
    for line in fin:
        strs = line.split()
        if strs[1] != 'Res1' and strs[1] != 'nonzero' and float(strs[4]) <= 5.000:
            ind1 = cid_list.index(strs[1][2:])
            ind2 = cid_list.index(strs[2][2:])
            ene = _ene_from_strs(strs, scheme)
            ene_res[ind1] += 0.5 * ene
            ene_res[ind2] += 0.5 * ene
    for i in range(reslen):
        for j in range(reslen):
            mat[i][j] = ene_res[i] + ene_res[j]
            mat[j][i] = ene_res[i] + ene_res[j]
    return mat


def decoy_stat(cid_list, contact_map, decoy_num, sep, scheme, lig_pos):
    reslen = len(cid_list)
    mat_all = np.zeros((reslen, reslen, decoy_num))
    bad_seq = 0
    good_seq = 0
    for i in range(decoy_num):
        temp = read_log(str(i + 1) + '.log', cid_list, scheme)
        if temp.sum() == 0:
            bad_seq = bad_seq + 1
        else:
            good_seq = good_seq + 1
            mat_all[:, :, good_seq - 1] = temp
    print("The number of Bad Sequences is: " + str(bad_seq))
    print("The number of Good Sequences is: " + str(good_seq))

    pro_mat = []
    res_mean = [0 for i in range(reslen)]
    res_std = [0 for i in range(reslen)]
    lig_mean = []
    lig_std = []
    for i in range(reslen):
        for j in range(i, reslen):
            if (lig_pos[i] == 0 and lig_pos[j] == 0) and (abs(i - j) > sep or cid_list[i][0] != cid_list[j][0]) and contact_map[i, j] <= 10.0:
                for k in range(good_seq):
                    if mat_all[i, j, k] != 0.0:
                        pro_mat.append(mat_all[i, j, k])
    pro_mean = np.mean(pro_mat)
    pro_std = np.std(pro_mat)

    for i in range(reslen):
        if (lig_pos[i] == 0):
            res_mean[i] = pro_mean
            res_std[i] = pro_std

    for i in range(reslen):
        if lig_pos[i] == 1:
            lig_mat = []
            for j in range(reslen):
                if lig_pos[j] == 0 and contact_map[i, j] <= 10.0:
                    for k in range(good_seq):
                        if mat_all[i, j, k] != 0.0:
                            lig_mat.append(mat_all[i, j, k])
            lig_mean.append(np.mean(lig_mat))
            lig_std.append(np.std(lig_mat))
            res_mean[i] = np.mean(lig_mat)
            res_std[i] = np.std(lig_mat)

    return pro_mean, pro_std, lig_mean, lig_std, res_mean, res_std


def frust_map(mat_nat, contact_map, minvalue, maxvalue, sep, cid_list,
              lig_pos, res_mean, res_std, ca_atoms, res_list_name):
    reslen = len(cid_list)
    frust = np.zeros((reslen, reslen))
    frust_d = np.zeros((reslen, reslen))
    fdat = open('tertiary_frustration.dat', 'w')
    for i in range(reslen):
        for j in range(i, reslen):
            if (lig_pos[i] == 0 and lig_pos[j] == 0) and (abs(i - j) > sep or cid_list[i][0] != cid_list[j][0]) and contact_map[i, j] <= 10.0 and mat_nat[i, j] != 999:
                frust[i, j] = (mat_nat[i, j] - res_mean[i]) / (res_std[i])
                frust[j, i] = (mat_nat[i, j] - res_mean[i]) / (res_std[i])
                fdat.write(str(i) + ' ' + str(j) + ' ' + cid_list[i][0] + ' ' + cid_list[j][0] + ' ' + str(ca_atoms[i][0]) + ' ' + str(ca_atoms[i][1]) + ' ' + str(ca_atoms[i][2]) + ' ' + str(ca_atoms[j][0]) + ' ' + str(ca_atoms[j][1]) + ' ' + str(ca_atoms[j][2]) + ' ' + str(contact_map[i, j]) + ' ' + res_list_name[i] + ' ' + res_list_name[j] + ' ' + str(mat_nat[i, j]) + ' ' + str(res_mean[i]) + ' ' + str(res_std[i]) + '\n')
            if (lig_pos[i] == 1 and lig_pos[j] == 0) and contact_map[i, j] <= 10.0:
                frust[i, j] = (mat_nat[i, j] - res_mean[i]) / (res_std[i])
                frust[j, i] = (mat_nat[i, j] - res_mean[i]) / (res_std[i])
                fdat.write(str(i) + ' ' + str(j) + ' ' + cid_list[i][0] + ' ' + cid_list[j][0] + ' ' + str(ca_atoms[i][0]) + ' ' + str(ca_atoms[i][1]) + ' ' + str(ca_atoms[i][2]) + ' ' + str(ca_atoms[j][0]) + ' ' + str(ca_atoms[j][1]) + ' ' + str(ca_atoms[j][2]) + ' ' + str(contact_map[i, j]) + ' ' + res_list_name[i] + ' ' + res_list_name[j] + ' ' + str(mat_nat[i, j]) + ' ' + str(res_mean[i]) + ' ' + str(res_std[i]) + '\n')
            if (lig_pos[i] == 0 and lig_pos[j] == 1) and contact_map[i, j] <= 10.0:
                frust[i, j] = (mat_nat[i, j] - res_mean[j]) / (res_std[j])
                frust[j, i] = (mat_nat[i, j] - res_mean[j]) / (res_std[j])
                fdat.write(str(i) + ' ' + str(j) + ' ' + cid_list[i][0] + ' ' + cid_list[j][0] + ' ' + str(ca_atoms[i][0]) + ' ' + str(ca_atoms[i][1]) + ' ' + str(ca_atoms[i][2]) + ' ' + str(ca_atoms[j][0]) + ' ' + str(ca_atoms[j][1]) + ' ' + str(ca_atoms[j][2]) + ' ' + str(contact_map[i, j]) + ' ' + res_list_name[i] + ' ' + res_list_name[j] + ' ' + str(mat_nat[i, j]) + ' ' + str(res_mean[j]) + ' ' + str(res_std[j]) + '\n')
    fdat.close()
    return frust, frust_d


def main():
    reslen = int(sys.argv[1])           # unused for sizing (kept for CLI parity)
    minvalue = float(sys.argv[2])
    maxvalue = float(sys.argv[3])
    decoy_num = int(sys.argv[4])
    sep = int(sys.argv[5])
    enable = int(sys.argv[6])
    scheme = sys.argv[7]
    minvalue_l = float(sys.argv[8])
    maxvalue_l = float(sys.argv[9])

    print("the cutoff of minimal frustration is: " + str(minvalue))
    print("the cutoff of high frustration is: " + str(maxvalue))
    print("the number of decoys used is: " + str(decoy_num))
    print("the sequence separation used is: " + str(sep))
    print("the scheme of " + str(scheme) + " frustratometer is used")

    contact_map, cid_list, atom_list, lig_pos, res_list_name, ca_atoms = get_index('3gso')
    mat_nat = read_nat_log('./native.log', cid_list, scheme)
    stat_mean, stat_std, lig_mean, lig_std, res_mean, res_std = decoy_stat(
        cid_list, contact_map, decoy_num, sep, scheme, lig_pos)
    print("protein decoy mean/std:", stat_mean, stat_std)
    frust, frust_d = frust_map(
        mat_nat, contact_map, minvalue, maxvalue, sep, cid_list,
        lig_pos, res_mean, res_std, ca_atoms, res_list_name)
    print("wrote tertiary_frustration.dat (" + str(len(cid_list)) + " residues)")


if __name__ == "__main__":
    main()
