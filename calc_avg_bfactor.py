def calculate_average_bfactor(pdb_file):
    total_bfactor = 0.0
    atom_count = 0

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    bfactor = float(line[60:66])
                    total_bfactor += bfactor
                    atom_count += 1
                except ValueError:
                    continue

    if atom_count == 0:
        return 0.0

    average_bfactor = total_bfactor / atom_count
    return average_bfactor


if __name__ == "__main__":
    pdb_file = "rfah_L142K_I146D_mask_mutate_94a41_model_4_ptm_r6_seed_010_mask_false_mut_L142K_I146D_id_X.pdb"
    avg_bfactor = calculate_average_bfactor(pdb_file)
    print(f"Average B-factor: {avg_bfactor:.2f}")
