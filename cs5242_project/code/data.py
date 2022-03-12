import numpy as np
import pandas as pd


def read_pdb(filename: str):
    """Read a protein file to get four atom information lists.

    You can copy this function to your project code.
    """
    with open(filename, "r") as file:
        strline_L = file.readlines()
    strline_L = [strline.strip() for strline in strline_L]

    X_list = [float(strline.split()[-3]) for strline in strline_L]
    Y_list = [float(strline.split()[-2]) for strline in strline_L]
    Z_list = [float(strline.split()[-1]) for strline in strline_L]
    atomtype_list = [strline.split()[-7][0] for strline in strline_L]

    return X_list, Y_list, Z_list, atomtype_list


def generate_negative_example(df_pairs, df_ligands, ratio, seed):
    proteins_ls = list(df_pairs["PID"])
    ligands_ls = list(df_ligands["LID"])
    np.random.seed(seed)

    out_proteins_ls = []
    out_ligands_ls = []
    target_ls = []

    for i in proteins_ls:
        paired_ligand = df_pairs[df_pairs["PID"] == i]["LID"].values[0]
        for j in range(ratio):
            out_proteins_ls.append(i)
            chosen_ligand = np.random.choice(
                [k for k in ligands_ls if k != paired_ligand], 1
            )[0]
            out_ligands_ls.append(chosen_ligand)
            target_ls.append(0)

    df_out = pd.DataFrame(
        {"PID": out_proteins_ls, "LID": out_ligands_ls, "target": target_ls}
    )

    return df_out
