import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import click

from code.model_v1 import MyNet
from code.model_v2 import MyProNet
from code.model_v3 import MyProInceptionNet
from code.MyDilatedNet import MyDilatedNet
from code.MyDilatedVoxelNet import MyDilatedVoxelNet
from code.model_utils import validation_result
from code.feature import (
    one_hot_protein,
    one_hot_smiles,
    process_protein_group,
    process_protein_group_onehot,
    process_protein_group_onehot_dist,
    process_protein_voxel,
)
from code.data import generate_negative_example, read_pdb, make_data

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)


class CustomDataset(Dataset):
    def __init__(self, df_pair, df_ligands, path):
        self.df_pair = df_pair
        self.df_ligands = df_ligands
        self.path = path

    def __len__(self):
        return len(self.df_pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid = self.df_pair["PID"][idx]
        lid = self.df_pair["LID"][idx]
        target = np.array([self.df_pair["target"][idx]])

        out_ligand = one_hot_smiles(
            self.df_ligands[self.df_ligands["LID"] == lid]["Smiles"].values[0]
        )
        X_list, Y_list, Z_list, atomtype_list = read_pdb(f"{self.path}/{pid}.pdb")
        out_protein = one_hot_protein(atomtype_list)
        return out_ligand, out_protein, target


class CustomDataset_v2(Dataset):
    def __init__(self, df_pair, df_ligands, df_centroids, path):
        self.df_pair = df_pair
        self.df_ligands = df_ligands
        self.df_centroids = df_centroids
        self.path = path

    def __len__(self):
        return len(self.df_pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid = self.df_pair["PID"][idx]
        lid = self.df_pair["LID"][idx]
        target = np.array([self.df_pair["target"][idx]])

        out_ligand = one_hot_smiles(
            self.df_ligands[self.df_ligands["LID"] == lid]["Smiles"].values[0]
        )
        out_ligand = out_ligand.reshape(200, 67)
        out_protein = process_protein_group(pid, self.path, self.df_centroids)
        out_protein = out_protein.reshape(540, 5)
        return out_ligand, out_protein, target


class CustomDataset_v3(Dataset):
    def __init__(
        self, df_pair, df_ligands, df_centroids, path, distance=False, sort=False
    ):
        self.df_pair = df_pair
        self.df_ligands = df_ligands
        self.df_centroids = df_centroids
        self.path = path
        self.distance = distance
        self.sort = sort

    def __len__(self):
        return len(self.df_pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid = self.df_pair["PID"][idx]
        lid = self.df_pair["LID"][idx]
        target = np.array([self.df_pair["target"][idx]])

        out_ligand = one_hot_smiles(
            self.df_ligands[self.df_ligands["LID"] == lid]["Smiles"].values[0]
        )

        out_ligand = out_ligand.reshape(200, -1)
        if self.distance and self.sort:
            out_protein = process_protein_group_onehot_dist(
                pid, self.path, self.df_centroids, sort=True
            )
        elif self.distance and not self.sort:
            out_protein = process_protein_group_onehot_dist(
                pid, self.path, self.df_centroids, sort=False
            )
        elif not self.sort and not self.distance:
            out_protein = process_protein_group_onehot(pid, self.path)

        out_protein = out_protein.reshape(540, -1)
        return out_ligand, out_protein, target


class CustomDataset_v4(Dataset):
    def __init__(
        self, df_pair, df_ligands, df_centroids, path, distance=False, sort=False
    ):
        self.df_pair = df_pair
        self.df_ligands = df_ligands
        self.df_centroids = df_centroids
        self.path = path
        self.distance = distance
        self.sort = sort

    def __len__(self):
        return len(self.df_pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid = self.df_pair["PID"][idx]
        lid = self.df_pair["LID"][idx]
        target = np.array([self.df_pair["target"][idx]])

        out_ligand = one_hot_smiles(
            self.df_ligands[self.df_ligands["LID"] == lid]["Smiles"].values[0]
        )
        out_ligand = out_ligand.reshape(200, -1)

        out_protein = process_protein_voxel(
            pid, self.path, MAX_DIST=20, GRID_RESOLUTION=1
        )

        return out_ligand, out_protein, target


def train(
    name,
    df_train,
    df_test,
    df_ligands,
    df_centroids,
    learning_rate,
    max_learning_rate,
    num_epoch,
    batch_size,
    dropout_alpha,
    data_path,
    model_path,
    save_best_epoch,
    distance=False,
    sort=False,
):
    pdb_path = os.path.join(data_path, "pdbs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "mynet":
        train_dataset = CustomDataset(df_train, df_ligands, pdb_path)
        trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        test_dataset = CustomDataset(df_test, df_ligands, pdb_path)
        testloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        model = MyNet(dropout_alpha, device).to(device)
    elif name == "mypronet":
        train_dataset = CustomDataset_v2(df_train, df_ligands, df_centroids, pdb_path)
        trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        test_dataset = CustomDataset_v2(df_test, df_ligands, df_centroids, pdb_path)
        testloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        model = MyProNet().to(device)
    elif name == "myinceptionnet":
        train_dataset = CustomDataset_v2(df_train, df_ligands, df_centroids, pdb_path)
        trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        test_dataset = CustomDataset_v2(df_test, df_ligands, df_centroids, pdb_path)
        testloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        model = MyProInceptionNet().to(device)
    elif name == "mydilatednet":
        train_dataset = CustomDataset_v3(
            df_train, df_ligands, df_centroids, pdb_path, distance=distance, sort=sort
        )
        trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        test_dataset = CustomDataset_v3(
            df_test, df_ligands, df_centroids, pdb_path, distance=distance, sort=sort
        )
        testloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        model = MyDilatedNet().to(device)
    elif name == "mydilatedvoxelnet":
        train_dataset = CustomDataset_v4(df_train, df_ligands, df_centroids, pdb_path)
        trainloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        test_dataset = CustomDataset_v4(df_test, df_ligands, df_centroids, pdb_path)
        testloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        model = MyDilatedVoxelNet().to(device)
    else:
        raise NotImplementedError

    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        epochs=num_epoch,
        steps_per_epoch=len(trainloader),
    )

    best_val_loss = 100000000
    loss_ls = []
    loss_validation_ls = []
    for epoch in tqdm(range(num_epoch)):
        for i, data in enumerate(trainloader, 0):
            print(f"training on index {i}")
            ligand = data[0].to(device)
            protein = data[1].to(device)
            target = data[2].to(device).float()

            optimizer.zero_grad()
            outputs = model(protein.float(), ligand.float())
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            print("epoch {:3d} | {:5d} batches loss: {:.4f}".format(epoch, i + 1, loss))
            loss_ls.append(loss.detach().numpy())

        result = validation_result(testloader, model, device)
        loss_validation_ls.append(result)
        # use logloss as the benchmark
        if result["logloss"] < best_val_loss and epoch >= save_best_epoch:
            torch.save(model.state_dict(), f"{model_path}/best_model_epoch_{epoch}.pt")
            best_val_loss = result["logloss"]

    print("Finished training!")
    torch.save(model.state_dict(), f"{model_path}/{name}_model_dict.pt")
    torch.save(model, f"{model_path}/{name}_model.pt")
    df_loss = pd.DataFrame({"loss": loss_ls})
    df_loss.to_csv(f"{model_path}/loss_records.csv")
    df_loss = pd.DataFrame({"loss": loss_validation_ls})
    df_loss.to_csv(f"{model_path}/loss_validation_records.csv")


@click.command()
@click.argument("name", type=str)
@click.argument("model_path_name", type=str)
@click.argument("num_epoch", type=int)
@click.argument("distance", type=bool, required=False)
@click.argument("sort", type=bool, required=False)
def main(name, model_path_name, num_epoch, distance=False, sort=False):
    print("--------- reading data ---------")
    data_path = os.path.join(project_path, "dataset_20220217_2")
    df_train = pd.read_csv(os.path.join(data_path, "train_neg2_perct0.8.csv"))
    df_validation = pd.read_csv(os.path.join(data_path, "validation_neg2_perct0.1.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "test_neg2_perct0.1.csv"))
    # df_train = df_train.sample(frac=1).reset_index(drop=True)

    print("--------- training ---------")
    # num_epoch = 1  # 300
    batch_size = 128
    dropout_alpha = 0.5
    learning_rate = 0.0001
    max_learning_rate = 5e-3
    model_path = os.path.join(project_path, "output", model_path_name)  # to be updated
    df_ligands = pd.read_csv(os.path.join(data_path, "ligand.csv"))
    df_centriods = pd.read_csv(os.path.join(data_path, "centroids.csv"))
    save_best_epoch = 5
    train(
        name=name,
        df_train=df_train,
        df_test=df_validation,
        df_ligands=df_ligands,
        df_centroids=df_centriods,
        learning_rate=learning_rate,
        max_learning_rate=max_learning_rate,
        num_epoch=num_epoch,
        batch_size=batch_size,
        dropout_alpha=dropout_alpha,
        data_path=data_path,
        model_path=model_path,
        save_best_epoch=save_best_epoch,
        distance=distance,
        sort=sort,
    )


if __name__ == "__main__":
    main()
