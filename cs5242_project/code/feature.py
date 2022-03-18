from itertools import chain
import numpy as np


SMISET = {
    "C": 67,
    "l": 1,
    ".": 2,
    "c": 3,
    "1": 4,
    "2": 5,
    "(": 6,
    "N": 7,
    "=": 8,
    "3": 9,
    ")": 10,
    "n": 11,
    "[": 12,
    "H": 13,
    "]": 14,
    "O": 15,
    "@": 16,
    "s": 17,
    "+": 18,
    "/": 19,
    "S": 20,
    "F": 21,
    "-": 22,
    "4": 23,
    "B": 24,
    "r": 25,
    "o": 26,
    "\\": 27,
    "#": 28,
    "5": 29,
    "a": 30,
    "P": 31,
    "e": 32,
    "6": 33,
    "7": 34,
    "I": 35,
    "A": 36,
    "i": 37,
    "8": 38,
    "9": 39,
    "Z": 40,
    "K": 41,
    "L": 42,
    "%": 43,
    "0": 44,
    "T": 45,
    "g": 46,
    "G": 47,
    "d": 48,
    "M": 49,
    "b": 50,
    "u": 51,
    "t": 52,
    "R": 53,
    "p": 54,
    "m": 55,
    "W": 56,
    "Y": 57,
    "V": 58,
    "~": 59,
    "U": 60,
    "E": 61,
    "f": 62,
    "X": 63,
    "D": 64,
    "y": 65,
    "h": 66,
}

PROTSET = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "O": 21,
}

AMINOSET = {
    "DG": 1,
    "O3'": 2,
    "N7": 3,
    "C": 4,
    "C3'": 5,
    "TYR": 6,
    "C4": 7,
    "MET": 8,
    "CE3": 9,
    "SER": 10,
    "N": 11,
    "DA": 12,
    "OG": 13,
    "CZ": 14,
    "C8": 15,
    "U": 16,
    "OP1": 17,
    "ND2": 18,
    "CG2": 19,
    "N4": 20,
    "C2": 21,
    "CH2": 22,
    "C5": 23,
    "CB": 24,
    "P": 25,
    "GLY": 26,
    "OE2": 27,
    "N2": 28,
    "UNK": 29,
    "CD1": 30,
    "ASP": 31,
    "C2'": 32,
    "N6": 33,
    "DT": 34,
    "DC": 35,
    "ND1": 36,
    "O": 37,
    "C4'": 38,
    "OD1": 39,
    "CD": 40,
    "CE1": 41,
    "CYS": 42,
    "DU": 43,
    "ARG": 44,
    "HIS": 45,
    "LYS": 46,
    "OH": 47,
    "GLN": 48,
    "CE2": 49,
    "NH1": 50,
    "NE2": 51,
    "OXT": 52,
    "NZ": 53,
    "G": 54,
    "OG1": 55,
    "VAL": 56,
    "C1'": 57,
    "O5'": 58,
    "PRO": 59,
    "CA": 60,
    "SD": 61,
    "N9": 62,
    "C5'": 63,
    "PHE": 64,
    "NE": 65,
    "O6": 66,
    "O4'": 67,
    "A": 68,
    "CE": 69,
    "O2'": 70,
    "CD2": 71,
    "SG": 72,
    "C6": 73,
    "OP2": 74,
    "CZ2": 75,
    "CZ3": 76,
    "N3": 77,
    "N1": 78,
    "OE1": 79,
    "THR": 80,
    "CG1": 81,
    "ASN": 82,
    "O2": 83,
    "OD2": 84,
    "ALA": 85,
    "ILE": 86,
    "TRP": 87,
    "CG": 88,
    "NH2": 89,
    "LEU": 90,
    "GLU": 91,
    "O4": 92,
    "NE1": 93,
}

CHAINSET = {
    "E": 1,
    "HIS": 2,
    "C": 3,
    "GLY": 4,
    "LYS": 5,
    "TYR": 6,
    "GLN": 7,
    "D": 8,
    "G": 9,
    "THR": 10,
    "B": 11,
    "MET": 12,
    "ASP": 13,
    "VAL": 14,
    "ASN": 15,
    "SER": 16,
    "PRO": 17,
    "ALA": 18,
    "ILE": 19,
    "TRP": 20,
    "PHE": 21,
    "H": 22,
    "U": 23,
    "L": 24,
    "LEU": 25,
    "A": 26,
    "GLU": 27,
    "CYS": 28,
    "ARG": 29,
}


def one_hot_smiles(line, MAX_SMI_LEN=200):
    X = np.zeros((1, MAX_SMI_LEN, len(SMISET)))  # +1

    if type(line) != str:
        print("SMILE format is not str!")
    line = line.strip()
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        tmp = SMISET.get(ch)
        if tmp:
            X[0, i, tmp - 1] = 1
        else:
            print(line, "exits not in SMISET character", ch)
    return X


def one_hot_protein(line, MAX_SEQ_LEN=4300):
    X = np.zeros((1, MAX_SEQ_LEN, len(PROTSET)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        tmp = PROTSET.get(ch)
        if tmp:
            X[0, i, tmp - 1] = 1
        # else:
        #     print("exits not in PROTSET character", ch)
    return X


def one_hot_protein_group(line, type, MAX_SEQ_LEN=540):
    if type == "amino":
        X = np.zeros((MAX_SEQ_LEN, len(AMINOSET)))
    elif type == "chain":
        X = np.zeros((MAX_SEQ_LEN, len(CHAINSET)))

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        if type == "amino":
            tmp = AMINOSET.get(ch)
        elif type == "chain":
            tmp = CHAINSET.get(ch)
        if tmp:
            X[i, tmp - 1] = 1
        else:
            print("exits not in dict character", ch)
    return X


def convert_protein_list_to_matrix(X_list, Y_list, Z_list):
    protein_coords = np.concatenate(
        (
            np.array(X_list).reshape(-1, 1),
            np.array(Y_list).reshape(-1, 1),
            np.array(Z_list).reshape(-1, 1),
        ),
        axis=1,
    )
    return protein_coords


def read_pdb_group(filename: str):
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
    atomgroup_list = [strline.split()[-4] for strline in strline_L]
    aminogroup_list = [strline.split()[-6] for strline in strline_L]
    chain_list = [strline.split()[-5] for strline in strline_L]

    return (
        X_list,
        Y_list,
        Z_list,
        atomtype_list,
        atomgroup_list,
        aminogroup_list,
        chain_list,
    )


def process_protein_group_onehot(pid, pid_path):
    (
        X_list,
        Y_list,
        Z_list,
        atomtype_list,
        atomgroup_list,
        aminogroup_list,
        chain_list,
    ) = read_pdb_group(f"{pid_path}/{pid}.pdb")
    unique_group, group_index = np.unique(atomgroup_list, return_index=True)
    unique_amino_list = [aminogroup_list[i] for i in group_index]
    unique_chain_list = [chain_list[i] for i in group_index]

    amino_encode = one_hot_protein_group(unique_amino_list, "amino")
    chain_encode = one_hot_protein_group(unique_chain_list, "chain")

    combined_matrix = np.concatenate((amino_encode, chain_encode), axis=1)
    return combined_matrix


def process_protein_group_onehot_dist(
    pid, pid_path, df_centroids, sort=False, max_length=540
):
    (
        X_list,
        Y_list,
        Z_list,
        atomtype_list,
        atomgroup_list,
        aminogroup_list,
        chain_list,
    ) = read_pdb_group(f"{pid_path}/{pid}.pdb")
    unique_group, group_index = np.unique(atomgroup_list, return_index=True)
    unique_amino_list = [aminogroup_list[i] for i in group_index]
    unique_chain_list = [chain_list[i] for i in group_index]

    group_coords_X = [X_list[i] for i in group_index]
    group_coords_Y = [Y_list[i] for i in group_index]
    group_coords_Z = [Z_list[i] for i in group_index]
    amino_encode = one_hot_protein_group(unique_amino_list, "amino")
    chain_encode = one_hot_protein_group(unique_chain_list, "chain")

    protein_coords = convert_protein_list_to_matrix(
        group_coords_X, group_coords_Y, group_coords_Z
    )
    centroid_coordinate = df_centroids[df_centroids["PID"] == pid].iloc[0, 1:4].values
    euclidean_dist = np.sum(
        np.square(protein_coords - centroid_coordinate), axis=1
    ).reshape(-1, 1)
    euclidean_dist = np.pad(
        euclidean_dist,
        [(0, max_length - euclidean_dist.shape[0]), (0, 0)],
        mode="constant",
        constant_values=0,
    )

    combined_matrix = np.concatenate(
        (amino_encode, chain_encode, euclidean_dist), axis=1
    )
    if sort:
        combined_matrix = combined_matrix[combined_matrix[:, -1].argsort()]

    return combined_matrix


def process_protein_group(pid, pid_path, df_centroids, max_length=540):
    X_list, Y_list, Z_list, atomtype_list, atomgroup_list, _, _ = read_pdb_group(
        f"{pid_path}/{pid}.pdb"
    )
    unique_group, group_index = np.unique(atomgroup_list, return_index=True)
    group_coords_X = [X_list[i] for i in group_index]
    group_coords_Y = [Y_list[i] for i in group_index]
    group_coords_Z = [Z_list[i] for i in group_index]
    atomtype_list = [atomtype_list[i] for i in group_index]
    atomtype_list = [PROTSET[i] if i in PROTSET.keys() else -1 for i in atomtype_list]

    protein_coords = convert_protein_list_to_matrix(
        group_coords_X, group_coords_Y, group_coords_Z
    )
    centroid_coordinate = df_centroids[df_centroids["PID"] == pid].iloc[0, 1:4].values
    euclidean_dist = np.sum(np.square(protein_coords - centroid_coordinate), axis=1)

    combined_matrix = np.column_stack(
        (group_coords_X, group_coords_Y, group_coords_Z, atomtype_list, euclidean_dist)
    )

    if max_length - combined_matrix.shape[0] >= 0:
        combined_matrix_pad = np.pad(
            combined_matrix,
            [(0, max_length - combined_matrix.shape[0]), (0, 0)],
            mode="constant",
            constant_values=0,
        )
    else:
        combined_matrix_pad = combined_matrix[0:max_length, :]

    return combined_matrix_pad.astype(float)
