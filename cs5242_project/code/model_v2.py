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

        conv_individual = []
        ic = seq_embed_size
        for oc in [32, 64, 128]:
            conv_individual.append(nn.Conv1d(ic, oc, 3))
            conv_individual.append(nn.BatchNorm1d(oc))
            conv_individual.append(nn.ReLU())
            ic = oc
        conv_individual.append(nn.AdaptiveMaxPool1d(1))
        conv_individual.append(Squeeze())
        self.conv_individual = nn.Sequential(*conv_individual)

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
        print(f"protein input: {protein}")
        protein_embed = self.seq_embed_protein(protein)  # (N,L,32)
        print(f"protein embed: {protein_embed}")
        protein_embed = torch.transpose(protein_embed, 1, 2)  # (N,32,L)
        print(f"protein transpose: {protein_embed}")
        protein_conv = self.conv_individual(protein_embed)  # (N,128)
        print(f"protein conv: {protein_conv}")

        # assert pkt.shape == (N,L,43)
        ligand_embed = self.seq_embed_ligand(ligand)  # (N,L,32)
        # print(f"ligand embed: {ligand_embed}")
        ligand_embed = torch.transpose(ligand_embed, 1, 2)  # (N,32,L)
        # print(f"ligand transpose: {ligand_embed}")
        ligand_conv = self.conv_individual(ligand_embed)  # (N,128)
        # print(f"ligand conv: {ligand_conv}")

        cat = torch.cat([protein_conv, ligand_conv], dim=1)  # (N,128*2)
        cat = self.cat_dropout(cat)
        print(cat)

        output = self.classifier(cat)
        return output
