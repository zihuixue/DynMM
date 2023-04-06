import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from unimodals.common_models import MLP, MaxOut_MLP
from datasets.imdb.get_data import get_dataloader
from training_structures.unimodal import train, test


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("imdb",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--mod", type=int, default=0, help="0: text; 1: image")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='no training')
    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    modality = 'image' if args.mod == 1 else 'text'
    encoderfile = "./log/imdb/encoder_" + modality + ".pt"
    headfile = "./log/imdb/head_" + modality + ".pt"
    traindata, validdata, testdata = get_dataloader("./data/multimodal_imdb.hdf5", "./data/mmimdb", vgg=True, batch_size=128, no_robust=True)

    log1, log2 = [], []
    for n in range(args.n_runs):
        if args.mod == 0:
            encoders = MLP(300, 512, 512).cuda()
            head = MLP(512, 512, 23).cuda()
        else:
            encoders = MLP(4096, 1024, 512).cuda()
            head = MLP(512, 512, 23).cuda()

        if not args.eval_only:
            train(encoders, head, traindata, validdata, 1000, early_stop=True, task="multilabel",
                  save_encoder=encoderfile, save_head=headfile,
                  modalnum=args.mod, optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01,
                  criterion=torch.nn.BCEWithLogitsLoss())

        print(f"Testing model {encoderfile} and {headfile}:")
        encoder = torch.load(encoderfile).cuda()
        head = torch.load(headfile).cuda()

        tmp = test(encoder, head, testdata, "imdb", modality, task="multilabel", modalnum=args.mod, no_robust=True,
             measure_time=args.measure)
        log1.append(tmp['f1_micro'])
        log2.append(tmp['f1_macro'])

    print(log1, log2)
    print(f'Finish {args.n_runs} runs')
    print(f'f1 micro {np.mean(log1) * 100:.2f} ± {np.std(log1) * 100:.2f}')
    print(f'f1 macro {np.mean(log2) * 100:.2f} ± {np.std(log2) * 100:.2f}')