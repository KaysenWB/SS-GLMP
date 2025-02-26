import numpy as np
import torch.nn as nn
from pytorch_tcn import TCN
import torch
from torch_geometric.utils import dense_to_sparse



def Traj_loss(pred, target):
    #see = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    if pred.dim() == 4:
        target = target.unsqueeze(2).repeat(1,1,10,1)

    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=0)
    loss_traj = traj_rmse.mean()

    return loss_traj


class GcnLayer(nn.Module):
    def __init__(self, args, feats_in, feats_out):
        super(GcnLayer,self).__init__()
        self.weight = nn.Parameter(torch.randn(feats_in, feats_out))
        self.norm = nn.LayerNorm(feats_out)
        self.device = args.device
        self.relu = nn.ReLU()

    def forward(self, v, a ):
        a_ = a + torch.eye(a.size(0)).to(self.device)
        D = a_.sum(dim=1)
        D_= torch.diag(D.pow(-0.5))
        A = D_ @ a_ @ D_
        V = v @ self.weight
        V = A @ V
        return self.norm(self.relu(V))


class GLRP(nn.Module):
    def __init__(self, args):
        super(GLRP, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.pred_length = self.args.pred_length
        self.K = self.args.K

        self.gcnlater1 = GcnLayer(args,self.feats_hidden, self.feats_hidden)
        self.gcnlater2 = GcnLayer(args,self.feats_hidden, self.feats_hidden)

        self.pro1 = nn.Linear(self.obs_length, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro4 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro5 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro6 = nn.Linear(self.feats_hidden, self.pred_length)

        self.relu = nn.ReLU()
        self.layer = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=8)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=1)

        self.norm1 = nn.LayerNorm(self.feats_hidden)
        self.norm2 = nn.LayerNorm(self.feats_hidden)
        self.norm3 = nn.LayerNorm(self.feats_hidden)
        self.g_loss = nn.BCEWithLogitsLoss()


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
        adj = torch.cat([inputs[2][-1:]] * self.feats_in,dim=0)

        adj = self.block_diagonal(adj).to(self.device)

        mina = adj.min()
        maxa = adj.max()
        ADJ = (adj - mina)/(maxa-mina)

        batch_in = batch_in.permute(2,1,0)
        enc_ = self.pro2(self.relu(self.pro1(batch_in)))

        enc = enc_.view( -1, self.feats_hidden)

        z = self.norm1(self.gcnlater1(enc, ADJ) + enc)
        z = self.norm2(self.gcnlater2(z, ADJ)+ z).unsqueeze(1).repeat(1, self.K, 1)

        mu = self.pro3(z)
        log_var = self.pro4(z)
        Z = mu + torch.randn_like(log_var) * torch.exp(0.5 * log_var)
        out = self.norm3(self.relu(self.pro5(Z))+z)

        out = self.encoder(out)

        out = self.pro6(out.view(self.feats_in, -1, self.K, self.feats_hidden)).permute(3, 1, 2, 0)
        traj = out[-self.pred_length:, :, :]

        traj_loss = Traj_loss(traj, tar_y)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))

        Z = Z.mean(1)
        graph_loss = self.g_loss(Z @ Z.t(), ADJ)

        loss = traj_loss + kl_loss + graph_loss

        return traj, loss



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
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=6)
        self.layer2 = nn.TransformerDecoderLayer(d_model=self.feats_hidden,nhead=self.heads)
        self.decoder = nn.TransformerDecoder(self.layer2, num_layers=6)

        self.input_en1 = nn.Linear(self.feats_in, self.feats_hidden//2)
        self.input_en2 = nn.Linear(self.feats_hidden//2, self.feats_hidden)

        self.output_fc = nn.Linear(self.feats_hidden, self.feats_out)
        self.relu = nn.ReLU()


    def forward(self, inputs, iftrain):

        batch = inputs[0][:,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, : self.feats_out]
        his_enc = batch[:self.obs_length, :, :]

        his_enc = self.input_en2(self.relu(self.input_en1(his_enc)))
        enc = self.pro_len1(his_enc.permute(1, 2, 0))


        traj = self.encoder(enc)
        traj = self.decoder(enc,traj)
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
        self.TCN = TCN(self.obs_length, [512,self.obs_length], kernel_size=3)
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

