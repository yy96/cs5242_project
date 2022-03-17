import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import click

from model_v1 import MyNet
from model_v2 import MyProNet
from model_utils import validation_result
from feature import one_hot_protein, one_hot_smiles, process_protein_group
from data import generate_negative_example, read_pdb, make_data

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
        # print(out_ligand.shape)
        out_ligand = out_ligand.reshape(200, 67)
        out_protein = process_protein_group(pid, self.path, self.df_centroids)
        # print(out_protein.shape)
        out_protein = out_protein.reshape(540, 5)
        return out_ligand, out_protein, target


def train_mynet(
    df_train,
    df_test,
    df_ligands,
    learning_rate,
    max_learning_rate,
    num_epoch,
    batch_size,
    dropout_alpha,
    data_path,
    model_path,
    save_best_epoch,
):
    pdb_path = os.path.join(data_path, "pdbs")
    train_dataset = CustomDataset(df_train, df_ligands, pdb_path)
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_dataset = CustomDataset(df_test, df_ligands, pdb_path)
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyNet(dropout_alpha, device).to(device)
    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        epochs=num_epoch,
        steps_per_epoch=len(trainloader),
    )

    best_epoch = -1
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
            torch.save(model.state_dict(), f"{model_path}/best_model.pt")
            best_val_loss = result["logloss"]
            best_epoch = epoch

    print("Finished training!")
    torch.save(model.state_dict(), f"{model_path}/mynet_model_dict.pt")
    torch.save(model, f"{model_path}/mynet_model.pt")
    df_loss = pd.DataFrame({"loss": loss_ls})
    df_loss.to_csv(f"{model_path}/loss_records.csv")
    df_loss = pd.DataFrame({"loss": loss_validation_ls})
    df_loss.to_csv(f"{model_path}/loss_validation_records.csv")


def train_pronet(
    df_train,
    df_test,
    df_ligands,
    df_centroids,
    max_learning_rate,
    num_epoch,
    batch_size,
    dropout_alpha,
    data_path,
    model_path,
    save_best_epoch,
):
    pdb_path = os.path.join(data_path, "pdbs")
    train_dataset = CustomDataset_v2(df_train, df_ligands, df_centroids, pdb_path)
    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_dataset = CustomDataset_v2(df_test, df_ligands, df_centroids, pdb_path)
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyProNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.AdamW(model.parameters())
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=max_learning_rate,
    #     epochs=num_epoch,
    #     steps_per_epoch=len(trainloader),
    # )

    best_epoch = -1
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
            print(outputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            print("epoch {:3d} | {:5d} batches loss: {:.4f}".format(epoch, i + 1, loss))
            loss_ls.append(loss.detach().numpy())

        result = validation_result(testloader, model, device)
        loss_validation_ls.append(result)
        # use logloss as the benchmark
        if result["logloss"] < best_val_loss and epoch >= save_best_epoch:
            torch.save(model.state_dict(), f"{model_path}/best_model.pt")
            best_val_loss = result["logloss"]
            best_epoch = epoch

    print("Finished training!")
    torch.save(model.state_dict(), f"{model_path}/mynet_model_dict.pt")
    torch.save(model, f"{model_path}/mynet_model.pt")
    df_loss = pd.DataFrame({"loss": loss_ls})
    df_loss.to_csv(f"{model_path}/loss_records.csv")
    df_loss = pd.DataFrame({"loss": loss_validation_ls})
    df_loss.to_csv(f"{model_path}/loss_validation_records.csv")


@click.command()
@click.argument("model", type=str)
def main(model):
    print("--------- reading data ---------")
    data_path = os.path.join(project_path, "dataset_20220217_2")
    df_train = pd.read_csv(os.path.join(data_path, "train_neg2_perct0.8.csv"))
    df_validation = pd.read_csv(
        os.path.join(data_path, "validation_neg2_perct0.05.csv")
    )
    df_test = pd.read_csv(os.path.join(data_path, "test_neg2_perct0.1.csv"))
    # df_train = df_train.sample(frac=1).reset_index(drop=True)

    print("--------- training ---------")
    num_epoch = 60  # 300
    batch_size = 128
    dropout_alpha = 0.5
    learning_rate = 0.0001
    max_learning_rate = 5e-3
    model_path = os.path.join(project_path, "output", "mypronet_v1")  # to be updated
    df_ligands = pd.read_csv(os.path.join(data_path, "ligand.csv"))
    df_centriods = pd.read_csv(os.path.join(data_path, "centroids.csv"))
    save_best_epoch = 5
    if model == "mynet":
        train_mynet(
            df_train=df_train,
            df_test=df_validation,
            df_ligands=df_ligands,
            learning_rate=learning_rate,
            max_learning_rate=max_learning_rate,
            num_epoch=num_epoch,
            batch_size=batch_size,
            dropout_alpha=dropout_alpha,
            data_path=data_path,
            model_path=model_path,
            save_best_epoch=save_best_epoch,
        )
    elif model == "mypronet":
        train_pronet(
            df_train=df_train,
            df_test=df_validation,
            df_ligands=df_ligands,
            df_centroids=df_centriods,
            max_learning_rate=max_learning_rate,
            num_epoch=num_epoch,
            batch_size=batch_size,
            dropout_alpha=dropout_alpha,
            data_path=data_path,
            model_path=model_path,
            save_best_epoch=save_best_epoch,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
