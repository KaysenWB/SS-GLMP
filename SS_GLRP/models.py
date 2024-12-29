import numpy as np
import torch.nn as nn
from pytorch_tcn import TCN
import torch
from torch_geometric.nn.models import GCN, GraphSAGE,GAT,GIN,GAE,GraphUNet
from torch_geometric.utils import dense_to_sparse
from graph_vae import VGAE
import torch_geometric as pyg



def Traj_loss(pred, target):
    #see = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))

    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=0)
    loss_traj = traj_rmse.mean()

    return loss_traj



class GVAET(nn.Module):
    def __init__(self, args):
        super(GVAET, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.pred_length = self.args.pred_length

        self.en_laters = GCN(in_channels=self.feats_hidden, hidden_channels=self.feats_hidden,
                        num_layers=1, out_channels=self.feats_hidden)
        self.Gencoder = VGAE(self.en_laters, None)

        self.Gdecoder = GCN(in_channels=self.feats_hidden, hidden_channels=self.feats_hidden ,
                       num_layers=1, out_channels=self.feats_hidden)
        self.GdecoderL = nn.Linear(self.feats_hidden, self.feats_hidden)


        self.decoder = nn.Sequential(
            nn.Linear(self.feats_hidden, self.feats_hidden*2),
            nn.ReLU(),
            nn.Linear(self.feats_hidden*2, self.feats_hidden)
        )
        self.de1 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.de2 = nn.Linear(self.feats_hidden, self.feats_hidden)

        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.obs_length, self.feats_hidden)
        self.pro4 = nn.Linear(self.feats_hidden, self.obs_length)
        self.pro5 = nn.Linear(self.feats_hidden, self.feats_out)
        self.pro6 = nn.Linear(self.feats_hidden//2, self.feats_out)

        self.relu = nn.ReLU()
        self.layer = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=8)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=6)

        self.layer2 = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=8)
        self.encoder2 = nn.TransformerEncoder(self.layer, num_layers=1)

    def block_diagonal(self, matrices):

        shapes = [m.shape for m in matrices]

        total_rows = sum(shape[0] for shape in shapes)
        total_cols = sum(shape[1] for shape in shapes)

        result = torch.zeros(total_rows, total_cols)

        row_idx = 0
        col_idx = 0
        for matrix in matrices:
            h, w = matrix.shape
            result[row_idx:row_idx + h, col_idx:col_idx + w] = matrix
            row_idx += h
            col_idx += w

        return result

    def forward(self, inputs, iftrain):

        batch_in = inputs[0][:self.obs_length, :, :self.feats_in]
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]
        adj = inputs[2][:self.obs_length, :, :]
        adj = self.block_diagonal(adj).to(self.device)

        mina = adj.min()
        maxa = adj.max()
        ADJ = (adj - mina)/(maxa-mina)
        #ADJ[ADJ < 0.2] = 0
        edge_index, edge_attr = dense_to_sparse(ADJ)

        enc = self.pro2(self.relu(self.pro1(batch_in)))
        enc = enc.view(-1, self.feats_hidden)
        Z = self.Gencoder.encode(enc, edge_index, edge_attr)
        kld = self.Gencoder.kl_loss()
        out = self.GdecoderL(Z)

        struct_loss = nn.BCEWithLogitsLoss()(Z @ Z.t(), ADJ)
        out = out.view( self.obs_length, -1, self.feats_hidden)

        out = self.encoder(self.pro3(out.permute(1, 2, 0)))
        out = self.pro4(out).permute(2, 0, 1)

        tra = self.pro5(out)
        traj = tra[-self.pred_length:, :, :]

        traj_loss = Traj_loss(traj, tar_y)
        loss =  traj_loss + kld + struct_loss
        return traj, loss



class GT(nn.Module):
    def __init__(self, args):
        super(GT, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.pred_length = self.args.pred_length

        self.GCN = GCN(in_channels=self.feats_hidden, hidden_channels=self.feats_hidden*2,
                        num_layers=2, out_channels=self.feats_hidden)

        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.obs_length, self.feats_hidden)
        self.pro4 = nn.Linear(self.feats_hidden, self.obs_length)
        self.pro5 = nn.Linear(self.feats_hidden, self.feats_out)
        self.pro6 = nn.Linear(self.feats_hidden//2, self.feats_out)

        self.relu = nn.ReLU()
        self.layer = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=8)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=6)

        self.layer2 = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=8)
        self.encoder2 = nn.TransformerEncoder(self.layer, num_layers=1)


    def block_diagonal(self, matrices):

        shapes = [m.shape for m in matrices]

        total_rows = sum(shape[0] for shape in shapes)
        total_cols = sum(shape[1] for shape in shapes)

        result = torch.zeros(total_rows, total_cols)

        row_idx = 0
        col_idx = 0
        for matrix in matrices:
            h, w = matrix.shape
            result[row_idx:row_idx + h, col_idx:col_idx + w] = matrix
            row_idx += h
            col_idx += w

        return result


    def forward(self, inputs, iftrain):

        batch_in = inputs[0][:self.obs_length, :, :self.feats_in]
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]
        adj = inputs[2][:self.obs_length, :, :]
        adj = self.block_diagonal(adj).to(self.device)

        mina = adj.min()
        maxa = adj.max()
        ADJ = (adj - mina)/(maxa-mina)
        ADJ[ADJ < 0.2] = 0
        edge_index, edge_attr = dense_to_sparse(ADJ)

        enc = self.pro2(self.relu(self.pro1(batch_in)))
        enc = enc.view(-1, self.feats_hidden)
        out = self.GCN(enc, edge_index, edge_attr)
        out = out.view( self.obs_length, -1, self.feats_hidden)

        out = self.encoder(self.pro3(out.permute(1, 2, 0)))
        out = self.pro4(out).permute(2, 0, 1)
        #out = self.encoder2(out)

        tra = self.pro5(out)
        traj = tra[-self.pred_length:, :, :]

        traj_loss = Traj_loss(traj, tar_y)

        return traj, traj_loss


class Trans(nn.Module):
    def __init__(self, args):
        super(Trans, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.heads = self.args.heads
        self.nlayer = self.args.layers_en

        self.pro_len1 = nn.Linear(self.obs_length, self.feats_hidden)
        self.pro_len2 = nn.Linear(self.feats_hidden, self.obs_length)

        self.layer = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=self.heads)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=self.nlayer)

        self.input_en1 = nn.Linear(self.feats_in, self.feats_hidden//2)
        self.input_en2 = nn.Linear(self.feats_hidden//2, self.feats_hidden)

        self.output_fc = nn.Linear(self.feats_hidden, self.feats_out)
        self.relu = nn.ReLU()


    def forward(self, inputs, iftrain):

        batch = inputs[0][:,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, : self.feats_out]
        his_enc = batch[:self.obs_length, :, :]

        his_enc = self.input_en2(self.relu(self.input_en1(his_enc)))
        traj = self.pro_len1(his_enc.permute(1, 2, 0))
        traj = self.encoder(traj)
        traj = self.pro_len2(traj).permute(2, 0, 1)
        traj = self.output_fc(traj)

        traj_loss = Traj_loss(traj, tar_y)
        return traj, traj_loss



class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = self.args.obs_length

        self.lstm = nn.LSTM(self.feats_hidden, self.feats_hidden)
        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_hidden//2)
        self.pro4 = nn.Linear(self.feats_hidden//2, self.feats_out)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]#.permute(1, 0, 2).to(self.device)
        padding = torch.zeros_like(batch)
        batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])

        enc = self.pro1(batch_in)
        enc = self.relu(enc)
        enc = self.pro2(enc)
        out, _ = self.lstm(enc)
        out = self.pro3(out)
        tra = self.pro4(out) # b2
        # b1
        traj = tra[self.obs_length:, :, :]#.permute(1, 0, 2)#.unsqueeze(2).repeat(1, 1, 20, 1)

        traj_loss = Traj_loss(traj, tar_y)
        return traj, traj_loss



class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = self.args.obs_length

        self.gru = nn.GRU(self.feats_hidden, self.feats_hidden)
        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_hidden//2)
        self.pro4 = nn.Linear(self.feats_hidden//2, self.feats_out)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]#.permute(1, 0, 2).to(self.device)
        padding = torch.zeros_like(batch)
        batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])

        enc = self.pro1(batch_in)
        enc = self.relu(enc)
        enc = self.pro2(enc)
        out, _ = self.gru(enc)
        out = self.pro3(out)
        tra = self.pro4(out) # b2
        # b1
        traj = tra[self.obs_length:, :, :]#.permute(1, 0, 2)#.unsqueeze(2).repeat(1, 1, 20, 1)

        traj_loss = Traj_loss(traj, tar_y)
        return traj, traj_loss







class GCNnet(nn.Module):
    def __init__(self, args):
        super(GCNnet, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.pred_length = self.args.pred_length

        self.GCN = GCN(in_channels=self.feats_hidden, hidden_channels=self.feats_hidden*2,
                        num_layers=2, out_channels=self.feats_hidden)

        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.obs_length, self.feats_hidden)
        self.pro4 = nn.Linear(self.feats_hidden, self.obs_length)
        self.pro5 = nn.Linear(self.feats_hidden, self.feats_hidden//2)
        self.pro6 = nn.Linear(self.feats_hidden//2, self.feats_out)

        self.lstm1 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.lstm2 = nn.Linear(self.feats_hidden, self.feats_hidden)

        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch_in = inputs[0][:self.obs_length, :, :self.feats_in]
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]
        adj = inputs[2][:self.obs_length, :, :]

        mina = adj.min()
        maxa = adj.max()
        ADJ = (adj - mina)/(maxa-mina)
        ADJ[ADJ < 0.5] = 0
        edge_index, edge_attr = dense_to_sparse(ADJ)

        enc = self.pro2(self.relu(self.pro1(batch_in)))
        enc = enc.view(-1, self.feats_hidden)
        out = self.GCN(enc, edge_index, edge_attr)
        out = out.view( self.obs_length, -1, self.feats_hidden)

        out, _ = self.lstm(self.pro3(out.permute(2, 1, 0)))
        out = self.pro4(out).permute(2, 1, 0)
        out, _ = self.lstm2(out)

        tra = self.pro6(self.pro5(out))
        traj = tra[-self.pred_length:, :, :]

        traj_loss = Traj_loss(traj, tar_y)

        return traj, traj_loss




class Seq2Seq(nn.Module):

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden

        self.encoder_se = nn.LSTM(self.feats_hidden,self.feats_hidden,num_layers=1)
        self.decoder_se = nn.LSTM(self.feats_hidden, self.feats_hidden, num_layers=1)

        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_out)
        self.pro4 = nn.Linear(self.feats_hidden // 2, self.feats_out)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in]  # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]  # .permute(1, 0, 2).to(self.device)
        #padding = torch.zeros_like(batch)
        #batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])
        batch_in = batch[:self.obs_length, :, :]

        enc = self.pro2(self.relu(self.pro1(batch_in)))
        encout,state = self.encoder_se(enc)

        out,_ = self.decoder_se(enc,state)
        tra = self.pro3(out)
        traj_loss = Traj_loss(tra, tar_y)

        return tra, traj_loss


class T_CNN(nn.Module):

    def __init__(self, args):
        super(T_CNN, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden

        self.project1 = nn.Linear(self.feats_in, self.feats_hidden//2)
        self.project2 = nn.Linear(self.feats_hidden//2, self.feats_hidden)
        self.project3 = nn.Linear(self.feats_hidden, self.feats_out)
        self.TCN = TCN(self.obs_length, [512,256,self.obs_length], kernel_size=3)
        self.relu= nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in]  # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]
        batch_in = batch[:self.obs_length, :, :]
        out = self.project2(self.relu(self.project1(batch_in)))
        out = out.permute(1, 0, 2)
        out = self.TCN(out)

        out = out.permute(1, 0, 2)
        tra = self.project3(out)

        traj_loss = Traj_loss(tra, tar_y)

        return tra, traj_loss

