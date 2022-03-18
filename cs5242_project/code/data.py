import numpy as np
import pandas as pd
import os


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


def make_data(data_path, negative_ratio, train_ratio, valid_ratio, save=True):
    df_pair = pd.read_csv(os.path.join(data_path, "pair.csv"))
    df_ligands = pd.read_csv(os.path.join(data_path, "ligand.csv"))
    df_centroids = pd.read_csv(os.path.join(data_path, "centroids.csv"))

    # remove th pair that have NA inputs in centroids
    df_centroids = df_centroids[
        (~df_centroids["x"].isna())
        & (~df_centroids["y"].isna())
        & (~df_centroids["z"].isna())
    ]
    df_pair = df_pair.loc[df_pair["PID"].isin(df_centroids["PID"])]

    num_positive = len(df_pair)
    df_positive = df_pair.copy()
    df_positive["target"] = 1
    df_train_positive = df_positive.iloc[
        0 : int(np.floor(num_positive * train_ratio)), :
    ]
    df_validation_positive = df_positive.iloc[
        int(np.floor(num_positive * train_ratio)) : int(
            np.floor(num_positive * (train_ratio + valid_ratio))
        ),
        :,
    ]
    df_test_positive = df_positive.iloc[
        int(np.floor(num_positive * (train_ratio + valid_ratio))) :,
        :,
    ]
    df_train_negative = generate_negative_example(
        df_train_positive, df_ligands, negative_ratio, 0
    )
    df_validation_negative = generate_negative_example(
        df_validation_positive, df_ligands, negative_ratio, 0
    )
    df_test_negative = generate_negative_example(
        df_test_positive, df_ligands, negative_ratio, 0
    )

    df_train = pd.concat([df_train_positive, df_train_negative])
    df_validation = pd.concat([df_validation_positive, df_validation_negative])
    df_test = pd.concat([df_test_positive, df_test_negative])
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_validation = df_validation.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    if save:
        df_train.to_csv(f"{data_path}/train_neg{negative_ratio}_perct{train_ratio}.csv")
        df_validation.to_csv(
            f"{data_path}/validation_neg{negative_ratio}_perct{valid_ratio}.csv"
        )
        df_test.to_csv(
            f"{data_path}/test_neg{negative_ratio}_perct{round(1-train_ratio-valid_ratio, 1)}.csv"
        )

    return df_train, df_validation, df_test


if __name__ == "__main__":
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    data_path = os.path.join(project_path, "dataset_20220217_2")

    make_data(data_path, 2, 0.8, 0.1)
