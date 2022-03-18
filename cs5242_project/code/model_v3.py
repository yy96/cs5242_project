import torch.nn as nn
import torch

PT_FEATURE_SIZE = 5
LIGAND_FEATURE_SIZE = 67


class InceptionBlock1D(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv1d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm1d(c_out["1x1"]),
            act_fn(),
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv1d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm1d(c_red["3x3"]),
            act_fn(),
            nn.Conv1d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm1d(c_out["3x3"]),
            act_fn(),
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv1d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm1d(c_red["5x5"]),
            act_fn(),
            nn.Conv1d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm1d(c_out["5x5"]),
            act_fn(),
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm1d(c_out["max"]),
            act_fn(),
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


class MyProInceptionNet(nn.Module):
    def __init__(self):
        super().__init__()

        seq_embed_size = 64

        self.seq_embed_protein = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)
        self.seq_embed_ligand = nn.Linear(LIGAND_FEATURE_SIZE, seq_embed_size)

        self.protein_inception_blocks = nn.Sequential(
            InceptionBlock1D(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=nn.ReLU,
            ),
            nn.MaxPool1d(3, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock1D(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=nn.ReLU,
            ),
            nn.MaxPool1d(3, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock1D(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=nn.ReLU,
            ),
            nn.MaxPool1d(3, stride=2, padding=1),  # 16x16 => 8x8
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(8704, 1024),  # identify input_shape here
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.ligand_inception_blocks = nn.Sequential(
            InceptionBlock1D(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=nn.ReLU,
            ),
            nn.MaxPool1d(3, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock1D(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=nn.ReLU,
            ),
            nn.MaxPool1d(3, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock1D(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=nn.ReLU,
            ),
            nn.MaxPool1d(3, stride=2, padding=1),  # 16x16 => 8x8
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(3200, 1024),  # identify input_shape here
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.cat_dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, protein, ligand):
        # assert seq.shape == (N,L,43)
        protein_embed = self.seq_embed_protein(protein)  # (N,L,32)
        protein_embed = torch.transpose(protein_embed, 1, 2)  # (N,32,L)
        protein_conv = self.protein_inception_blocks(protein_embed)  # (N,128)

        # assert pkt.shape == (N,L,43)
        ligand_embed = self.seq_embed_ligand(ligand)  # (N,L,32)
        ligand_embed = torch.transpose(ligand_embed, 1, 2)  # (N,32,L)
        ligand_conv = self.ligand_inception_blocks(ligand_embed)  # (N,128)

        cat = torch.cat([protein_conv, ligand_conv], dim=-1)  # (N,128*2)
        cat = self.cat_dropout(cat)

        output = self.classifier(cat)
        return output
