import torch.nn as nn
import torch

PT_FEATURE_SIZE = 122
PT_FEATURE_DIST_SIZE = 123
CHAR_SMI_SET_LEN = 67


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d
        )

    def forward(self, input):
        output = self.conv(input)
        return output


class CDilated3d(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv3d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d
        )

    def forward(self, input):
        output = self.conv(input)
        return output


class DilatedParllelResidualBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv3d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm3d(n), nn.PReLU())
        self.d1 = CDilated3d(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated3d(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated3d(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated3d(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated3d(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.br2 = nn.Sequential(nn.BatchNorm3d(nOut), nn.PReLU())

        if nIn != nOut:
            #             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            #             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class MyDilatedVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128

        seq_oc = 128
        # pkt_oc = 128
        smi_oc = 128

        self.smi_embed = nn.LazyLinear(smi_embed_size)

        self.seq_embed = nn.LazyLinear(
            seq_embed_size
        )  # (N, *, H_{in}) -> (N, *, H_{out})

        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool3d(1))  # (N, oc)
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)

        self.cat_dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(seq_oc + smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq, smi):
        # assert seq.shape == (N,L,43)
        seq_embed = self.seq_embed(seq)  # (N,L,32)
        seq_embed = torch.transpose(seq_embed, 1, -1)  # (N,32,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(smi)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        cat = torch.cat([seq_conv, smi_conv], dim=-1)  # (N,128*3)
        cat = self.cat_dropout(cat)

        output = self.classifier(cat)
        return output
