import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from tqdm import tqdm

from data import generate_negative_example, read_pdb, make_data
from feature import one_hot_protein, one_hot_smiles


project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

# inception block
class Conv2dLayer(nn.Module):
    def __init__(
        self,
        out_channels,
        num_row,
        num_col,
        padding="same",
        strides=1,
        use_bias=False,
        in_channels=1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(num_row, num_col),
                stride=strides,
                padding=padding,
                bias=use_bias,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.net(x)
        return output


class InceptionBlock(nn.Module):
    def __init__(
        self,
        filters_1x1,
        filters_3x3_reduce,
        filters_3x3,
        filters_5x5_reduce,
        filters_5x5,
        filters_pool_proj,
        in_channels=1,
    ):
        super().__init__()
        self.layer0 = Conv2dLayer(filters_1x1, 1, 1, in_channels=in_channels)
        self.layer1 = Conv2dLayer(filters_3x3_reduce, 1, 1, in_channels=in_channels)
        self.layer2 = Conv2dLayer(filters_3x3, 3, 3, in_channels=filters_3x3_reduce)
        self.layer3 = Conv2dLayer(filters_5x5_reduce, 1, 1, in_channels=in_channels)
        self.layer4 = Conv2dLayer(filters_5x5, 3, 3, in_channels=filters_5x5_reduce)
        self.layer5 = Conv2dLayer(filters_5x5, 3, 3, in_channels=filters_5x5)
        self.layer6 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool_stride = 2
        self.kernel_size = 3
        self.layer7 = Conv2dLayer(filters_pool_proj, 1, 1, in_channels=in_channels)

    def forward(self, x):
        branch_0 = self.layer0(x)
        branch_1 = self.layer1(x)
        branch_1 = self.layer2(branch_1)
        branch_2 = self.layer3(x)
        branch_2 = self.layer4(branch_2)
        branch_2 = self.layer5(branch_2)
        num_col = x.shape[3]
        num_row = x.shape[2]
        col_padding = int(
            np.ceil(
                (
                    num_col * (self.max_pool_stride - 1)
                    - self.max_pool_stride
                    + self.kernel_size
                )
                / 2
            )
        )
        row_padding = int(
            np.ceil(
                (
                    num_row * (self.max_pool_stride - 1)
                    - self.max_pool_stride
                    + self.kernel_size
                )
                / 2
            )
        )
        x_padded = F.pad(x, (col_padding, col_padding, row_padding, row_padding))
        branch_3 = self.layer6(x_padded)
        branch_3 = self.layer7(branch_3)

        x_out = torch.cat([branch_0, branch_1, branch_2, branch_3], dim=1)
        return x_out


class InceptionBlockB(nn.Module):
    def __init__(
        self,
        filters_1x1,
        filters_5x5_reduce,
        filters_5x5,
        filters_7x7_reduce,
        filters_1x7,
        filters_7x1,
        filters_pool_proj,
        in_channels,
    ):
        super().__init__()
        self.layer0 = Conv2dLayer(filters_1x1, 1, 1, in_channels=in_channels)
        self.layer1 = Conv2dLayer(filters_7x7_reduce, 1, 1, in_channels=in_channels)
        self.layer2 = Conv2dLayer(filters_1x7, 1, 7, in_channels=filters_7x7_reduce)
        self.layer3 = Conv2dLayer(filters_7x1, 7, 1, in_channels=filters_1x7)
        self.layer4 = Conv2dLayer(filters_5x5_reduce, 1, 1, in_channels=in_channels)
        self.layer5 = Conv2dLayer(filters_5x5, 3, 3, in_channels=filters_5x5_reduce)
        self.layer6 = Conv2dLayer(filters_5x5, 3, 3, in_channels=filters_5x5)
        self.layer7 = nn.AvgPool2d(kernel_size=3, stride=1)
        self.avg_pool_stride = 1
        self.kernel_size = 3
        self.layer8 = Conv2dLayer(filters_pool_proj, 1, 1, in_channels=in_channels)

    def forward(self, x):
        branch_0 = self.layer0(x)
        branch_1 = self.layer1(x)
        branch_1 = self.layer2(branch_1)
        branch_1 = self.layer3(branch_1)
        branch_2 = self.layer4(x)
        branch_2 = self.layer5(branch_2)
        branch_2 = self.layer6(branch_2)
        num_col = x.shape[3]
        num_row = x.shape[2]
        col_padding = int(
            np.ceil(
                (
                    num_col * (self.avg_pool_stride - 1)
                    - self.avg_pool_stride
                    + self.kernel_size
                )
                / 2
            )
        )
        row_padding = int(
            np.ceil(
                (
                    num_row * (self.avg_pool_stride - 1)
                    - self.avg_pool_stride
                    + self.kernel_size
                )
                / 2
            )
        )
        x_padded = F.pad(x, (col_padding, col_padding, row_padding, row_padding))
        branch_3 = self.layer7(x_padded)
        branch_3 = self.layer8(branch_3)

        x_out = torch.cat([branch_0, branch_1, branch_2, branch_3], dim=1)

        return x_out


class SimpleBlock(nn.Module):
    def __init__(self, nb_filter, num_row, num_col):
        self.layer = Conv2dLayer(nb_filter, num_row, num_col)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer(x)
        return x


class MyNet(nn.Module):
    def __init__(
        self,
        alpha,
        device,
        pro_branch_switch1="inception_block",
        pro_branch_switch2="inception_block",
        pro_branch_switch3="inception_block_b",
        pro_add_attention=False,
        comp_branch_switch1="inception_block",
        comp_branch_switch2="inception_block",
        comp_branch_switch3="inception_block_b",
        comp_add_attention=False,
    ):
        super().__init__()
        self.pro_branch_switch1 = pro_branch_switch1
        self.pro_branch_switch2 = pro_branch_switch2
        self.pro_branch_switch3 = pro_branch_switch3
        self.pro_add_attention = pro_add_attention
        self.comp_branch_switch1 = comp_branch_switch1
        self.comp_branch_switch2 = comp_branch_switch2
        self.comp_branch_switch3 = comp_branch_switch3
        self.comp_add_attention = comp_add_attention
        self.alpha = alpha
        self.device = device
        self._create_network()
        self._init_params()

    def _create_network(self):
        protein_blocks = []
        if self.pro_branch_switch1 == "inception_block":
            protein_blocks.append(
                InceptionBlock(
                    filters_1x1=8,
                    filters_3x3_reduce=1,
                    filters_3x3=32,
                    filters_5x5_reduce=1,
                    filters_5x5=32,
                    filters_pool_proj=16,
                    in_channels=1,
                )
            )
        else:
            protein_blocks.append(SimpleBlock(nb_filter=32, num_row=3, num_col=3))
        protein_blocks.append(nn.MaxPool2d(kernel_size=3, stride=3))

        if self.pro_branch_switch2 == "inception_block":
            protein_blocks.append(
                InceptionBlock(
                    filters_1x1=16,
                    filters_3x3_reduce=16,
                    filters_3x3=64,
                    filters_5x5_reduce=16,
                    filters_5x5=64,
                    filters_pool_proj=32,
                    in_channels=88,
                )
            )
        else:
            protein_blocks.append(SimpleBlock(nb_filter=64, num_row=3, num_col=3))
        protein_blocks.append(nn.MaxPool2d(kernel_size=3, stride=3))

        if self.pro_branch_switch3 == "inception_block":
            protein_blocks.append(
                InceptionBlock(
                    filters_1x1=32,
                    filters_3x3_reduce=64,
                    filters_3x3=128,
                    filters_5x5_reduce=64,
                    filters_5x5=128,
                    filters_pool_proj=64,
                )
            )
        elif self.pro_branch_switch3 == "inception_block_b":
            protein_blocks.append(
                InceptionBlockB(
                    filters_1x1=32,
                    filters_5x5_reduce=64,
                    filters_5x5=128,
                    filters_7x7_reduce=64,
                    filters_1x7=128,
                    filters_7x1=128,
                    filters_pool_proj=64,
                    in_channels=176,
                )
            )
        else:
            protein_blocks.append(SimpleBlock(nb_filter=128, num_row=3, num_col=3))
        protein_blocks.append(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # modified from 3 to 2

        if self.pro_add_attention:
            pass
        else:
            protein_blocks.append(nn.Flatten(start_dim=1, end_dim=-1))
            protein_blocks.append(nn.Linear(23232, 1024))  # identify input_shape here
            protein_blocks.append(nn.ReLU())
        protein_blocks.append(nn.Dropout(self.alpha))
        self.protein_blocks = nn.Sequential(*protein_blocks)

        ligand_blocks = []
        if self.comp_branch_switch1 == "inception_block":
            ligand_blocks.append(
                InceptionBlock(
                    filters_1x1=8,
                    filters_3x3_reduce=1,
                    filters_3x3=16,
                    filters_5x5_reduce=1,
                    filters_5x5=16,
                    filters_pool_proj=16,
                    in_channels=1,
                )
            )
        else:
            ligand_blocks.append(SimpleBlock(nb_filter=32, num_row=3, num_col=3))
        ligand_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        if self.comp_branch_switch2 == "inception_block":
            ligand_blocks.append(
                InceptionBlock(
                    filters_1x1=16,
                    filters_3x3_reduce=16,
                    filters_3x3=64,
                    filters_5x5_reduce=16,
                    filters_5x5=64,
                    filters_pool_proj=32,
                    in_channels=56,
                )
            )
        else:
            ligand_blocks.append(SimpleBlock(nb_filter=64, num_row=3, num_col=3))
        ligand_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        if self.comp_branch_switch3 == "inception_block":
            ligand_blocks.append(
                InceptionBlock(
                    filters_1x1=32,
                    filters_3x3_reduce=32,
                    filters_3x3=128,
                    filters_5x5_reduce=32,
                    filters_5x5=128,
                    filters_pool_proj=32,
                )
            )
        elif self.comp_branch_switch3 == "inception_block_b":
            ligand_blocks.append(
                InceptionBlockB(
                    filters_1x1=32,
                    filters_5x5_reduce=32,
                    filters_5x5=128,
                    filters_7x7_reduce=32,
                    filters_1x7=128,
                    filters_7x1=128,
                    filters_pool_proj=32,
                    in_channels=176,
                )
            )
        else:
            ligand_blocks.append(SimpleBlock(nb_filter=128, num_row=3, num_col=3))
        ligand_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        if self.comp_add_attention:
            pass
        else:
            ligand_blocks.append(nn.Flatten(start_dim=1, end_dim=-1))
            ligand_blocks.append(nn.Linear(64000, 640))  # identify input_shape here
            ligand_blocks.append(nn.ReLU())
        ligand_blocks.append(nn.Dropout(self.alpha))
        self.ligand_blocks = nn.Sequential(*ligand_blocks)

        combined_blocks = []
        combined_blocks.append(nn.Linear(1664, 512))
        combined_blocks.append(nn.ReLU())
        combined_blocks.append(nn.Dropout(self.alpha))
        self.combined_blocks = nn.Sequential(*combined_blocks)

        self.fc_pro_ligand_1 = nn.Linear(512, 64)
        self.fc_pro_ligand_2 = nn.Linear(64, 1)
        self.fc_sigmoid = nn.Sigmoid()
        self.fc_relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, protein_input, ligand_input):
        """
        protein_input: shape(1200, num_encoding)
        ligand_input: shape(200, num_encoding)
        """

        protein_output = self.protein_blocks(protein_input)
        ligand_output = self.ligand_blocks(ligand_input)

        protein_ligand = torch.cat([protein_output, ligand_output], dim=1)
        protein_ligand_out = self.combined_blocks(protein_ligand)

        x = self.dropout1(protein_ligand_out)
        x = self.fc_pro_ligand_1(x)
        x = self.fc_relu(x)
        x = self.fc_pro_ligand_2(x)
        x1 = self.fc_sigmoid(x)

        x = self.dropout2(protein_ligand_out)
        x = self.fc_pro_ligand_1(x)
        x = self.fc_relu(x)
        x = self.fc_pro_ligand_2(x)
        x2 = self.fc_sigmoid(x)

        x = self.dropout3(protein_ligand_out)
        x = self.fc_pro_ligand_1(x)
        x = self.fc_relu(x)
        x = self.fc_pro_ligand_2(x)
        x3 = self.fc_sigmoid(x)

        x = self.dropout4(protein_ligand_out)
        x = self.fc_pro_ligand_1(x)
        x = self.fc_relu(x)
        x = self.fc_pro_ligand_2(x)
        x4 = self.fc_sigmoid(x)

        x = self.dropout5(protein_ligand_out)
        x = self.fc_pro_ligand_1(x)
        x = self.fc_relu(x)
        x = self.fc_pro_ligand_2(x)
        x5 = self.fc_sigmoid(x)

        x_combined = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = torch.mean(x_combined, dim=1)
        out = out.reshape([-1, 1])

        return out

    def process_PDB(self, pid, pdbs_dir):
        X_list, Y_list, Z_list, atomtype_list = read_pdb(pdbs_dir)
        return one_hot_protein(atomtype_list)

    def batch_process_SMILE(self, ligands):
        ligands_ls = []
        for i in ligands:
            ligands_ls.append(one_hot_smiles(i))
        c = torch.stack(ligands_ls, dim=0)
        return c

    def batch_extend(self, p, l):
        batch = l.shape[0]
        p_batch = p.repeat(batch, 1, 1, 1)
        return p_batch, l

    def inference(self, PID, pdbs_dir, centroid, LIDs, ligands):
        p = self.process_PDB(PID, pdbs_dir).to(self.device)
        l = self.batch_process_SMILE(ligands).to(self.device)
        p, l = self.batch_extend(p, l)
        return self.forward(p, l)


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


def train_mynet(
    df_train,
    learning_rate,
    num_epoch,
    batch_size,
    dropout_alpha,
    project_path,
    model_dict_path,
    model_path,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyNet(dropout_alpha, device).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pdb_path = os.path.join(data_path, "pdbs")
    custom_dataset = CustomDataset(df_train, df_ligands, pdb_path)
    trainloader = DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    loss_log = []
    for epoch in tqdm(range(num_epoch)):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print(f"training on index {i}")
            ligand = data[0].to(device)
            protein = data[1].to(device)
            target = data[2].to(device).float()

            optimizer.zero_grad()
            outputs = model(protein.float(), ligand.float())  # .long()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            running_loss += loss.item()
            # if (i + 1) % batch_size == 0:
            print(
                "epoch {:3d} | {:5d} batches loss: {:.4f}".format(
                    epoch, i + 1, running_loss / 128
                )
            )
            running_loss = 0.0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(project_path, f"mynet_checkpoint_{epoch}.pt"),
        )
    print("Finished training!")
    torch.save(model.state_dict(), model_dict_path)
    torch.save(model, model_path)
    return loss_log, model


if __name__ == "__main__":
    print("--------- reading data ---------")
    data_path = os.path.join(project_path, "dataset_20220217_2")
    df_train, df_test = make_data(data_path, 2, 0.8)
    df_train.to_csv(os.path.join(data_path, "train.csv"))
    df_test.to_csv(os.path.join(data_path, "test.csv"))
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    print("--------- training ---------")
    num_epoch = 1  # 300
    batch_size = 128
    dropout_alpha = 0.5
    learning_rate = 0.0001
    model_dict_path = os.path.join(project_path, "mynet_parameters_v1.pt")
    model_path = os.path.join(project_path, "mynet_v1.pt")
    train_mynet(
        df_train,
        learning_rate,
        num_epoch,
        batch_size,
        dropout_alpha,
        project_path,
        model_dict_path,
        model_path,
    )
