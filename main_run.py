import argparse
import os
import time

import pandas as pd

from processor import processor
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser( description='GCN-Transformer')

parser.add_argument('--data_root', default='./AIS_process/AIS_processed_com.cpkl', type=str)
parser.add_argument('--save_dir', default='./output/', help='Directory for saving caches and models.')

parser.add_argument('--train_model', default='LSTM', help='[GT, Trans, LSTM, GRU, Seq2Seq, TCNN]')
parser.add_argument('--data_rate', default=[8,0,2], type=list) # train, val, test
parser.add_argument('--load_model', default='best', type=str, help="load pretrained model for test or training")

parser.add_argument('--seq_length', default=64, type=int)
parser.add_argument('--obs_length', default=32, type=int)
parser.add_argument('--label_length', default=32, type=int)
parser.add_argument('--pred_length', default=32, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.0005, type=float)
parser.add_argument('--clip', default=1, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
# model_TS_transformer


parser.add_argument('--layers_en', default=2, type=int)
parser.add_argument('--layers_de', default=1, type=int)
parser.add_argument('--heads', default=8, type=int, help='the number of heads in the multihead-attention models')

parser.add_argument('--device', default='cuda:2', type=str)
# Glow
parser.add_argument('--feats_in', default=5, type=int)
parser.add_argument('--feats_hidden', default=256, type=int)
parser.add_argument('--feats_out', default=2, type=int)


args = parser.parse_args()



model = ['GT', 'Trans', 'LSTM', 'GRU', 'Seq2Seq', 'TCNN']
#model = ['GT']
name =  args.save_dir
ADE_list = []
FDE_list = []
Time_list = []
infer_list =[]
for m in model:
    args.train_model = m


    args.save_dir = name + args.train_model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    trainer = processor(args)

    ts = time.time()
    trainer.train()
    train_time = time.time()-ts

    ADE, FDE, infer_time = trainer.test()
    print('train_time: {}-----------------'.format(train_time))
    print('infer_time: {}-----------------'.format(infer_time))

    trainer.pred()

    ADE_list.append(ADE)
    FDE_list.append(FDE)
    Time_list.append(train_time)
    infer_list.append(infer_time)

save_m = pd.DataFrame()
save_m['model'] = model
save_m['ADE'] = ADE_list
save_m['FDE'] = FDE_list
save_m['time'] = Time_list
save_m['infer'] = infer_list
table = open(name + 'Table','w')
save_m.to_csv(table)
table.close()
