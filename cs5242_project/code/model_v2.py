import torch.nn as nn
import torch

PT_FEATURE_SIZE = 5
LIGAND_FEATURE_SIZE = 67


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class MyProNet(nn.Module):
    def __init__(self):
        super().__init__()

        seq_embed_size = 128

        self.seq_embed_protein = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)
        self.seq_embed_ligand = nn.Linear(LIGAND_FEATURE_SIZE, seq_embed_size)

        conv_protein = []
        ic = seq_embed_size
        for oc in [32, 64, 128]:
            conv_protein.append(nn.Conv1d(ic, oc, 3))
            conv_protein.append(nn.BatchNorm1d(oc))
            conv_protein.append(nn.ReLU())
            ic = oc
        conv_protein.append(nn.AdaptiveMaxPool1d(1))
        conv_protein.append(Squeeze())
        self.conv_protein = nn.Sequential(*conv_protein)

        # self.conv1 = nn.Conv1d(ic, 32, 3)
        # self.conv2 = nn.Conv1d(32, 64, 3)
        # self.conv3 = nn.Conv1d(64, 128, 3)
        # self.batchnorm1 = nn.BatchNorm1d(32)
        # self.batchnorm2 = nn.BatchNorm1d(64)
        # self.batchnorm3 = nn.BatchNorm1d(128)
        # self.relu = nn.ReLU()
        # self.maxpool = nn.AdaptiveMaxPool1d(1)
        # self.squeeze = Squeeze()

        conv_ligand = []
        ic = seq_embed_size
        for oc in [32, 64, 128]:
            conv_ligand.append(nn.Conv1d(ic, oc, 3))
            conv_ligand.append(nn.BatchNorm1d(oc))
            conv_ligand.append(nn.ReLU())
            ic = oc
        conv_ligand.append(nn.AdaptiveMaxPool1d(1))
        conv_ligand.append(Squeeze())
        self.conv_ligand = nn.Sequential(*conv_ligand)

        self.cat_dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(seq_embed_size + seq_embed_size, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, protein, ligand):
        # assert seq.shape == (N,L,43)
        # print(f"protein input: {protein}")
        protein_embed = self.seq_embed_protein(protein)  # (N,L,32)
        # print(f"protein embed: {protein_embed}")
        protein_embed = torch.transpose(protein_embed, 1, 2)  # (N,32,L)
        # print(f"protein transpose: {protein_embed}")
        protein_conv = self.conv_protein(protein_embed)  # (N,128)
        # protein_conv = self.conv1(protein_embed)
        # print(f"protein conv1: {protein_conv}")
        # protein_conv = self.batchnorm1(protein_conv)
        # print(f"---- mean: {torch.mean(protein_conv)} ----")
        # print(f"---- var: {torch.var(protein_conv)} ----")
        # print(f"batch norm 1: {protein_conv}")
        # protein_conv = self.relu(protein_conv)
        # print(f"relu1: {protein_conv}")
        # protein_conv = self.conv2(protein_conv)
        # print(f"protein conv2: {protein_conv}")
        # protein_conv = self.batchnorm2(protein_conv)
        # protein_conv = self.relu(protein_conv)
        # protein_conv = self.conv3(protein_conv)
        # print(f"protein con3: {protein_conv}")
        # protein_conv = self.batchnorm3(protein_conv)
        # protein_conv = self.relu(protein_conv)
        # protein_conv = self.maxpool(protein_conv)
        # print(f"protein maxpool: {protein_conv}")
        # protein_conv = self.squeeze(protein_conv)
        # print(f"protein squeeze: {protein_conv}")

        # assert pkt.shape == (N,L,43)
        ligand_embed = self.seq_embed_ligand(ligand)  # (N,L,32)
        # print(f"ligand embed: {ligand_embed}")
        ligand_embed = torch.transpose(ligand_embed, 1, 2)  # (N,32,L)
        # print(f"ligand transpose: {ligand_embed}")
        ligand_conv = self.conv_ligand(ligand_embed)  # (N,128)

        cat = torch.cat([protein_conv, ligand_conv], dim=-1)  # (N,128*2)
        cat = self.cat_dropout(cat)

        output = self.classifier(cat)
        return output
