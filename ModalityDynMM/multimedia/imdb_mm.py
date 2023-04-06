import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch

from unimodals.common_models import Identity, Linear, MaxOut_MLP
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat, LowRankTensorFusion, MultiplicativeInteractions2Modal
from training_structures.Supervised_Learning import train, test


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("imdb", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--fuse", type=int, default=0, help="fusion model")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='no training')
    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    fusion_dict = {0: 'ef', 1: 'lf', 2: 'lrtf', 3: 'mim'}
    filename = "./log/imdb/best_" + fusion_dict[args.fuse] + ".pt"
    traindata, validdata, testdata = get_dataloader("./data/multimodal_imdb.hdf5", "./data/mmimdb", vgg=True, batch_size=128, no_robust=True)

    log1, log2 = [], []
    for n in range(args.n_runs):
        if args.fuse == 0:
            encoders = [Identity(), Identity()]
            head = MaxOut_MLP(23, 512, 4396).cuda()
            fusion = Concat().cuda()
            lr = 4e-2

        elif args.fuse == 1:
            encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
            head = Linear(1024, 23).cuda()
            fusion = Concat().cuda()
            lr = 8e-3

        elif args.fuse == 2:
            encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
            head = Linear(512, 23).cuda()
            fusion = LowRankTensorFusion([512, 512], 512, 128).cuda()
            lr = 8e-3

        else:
            encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
            head = Linear(1024, 23).cuda()
            fusion = MultiplicativeInteractions2Modal([512, 512], 1024, 'matrix').cuda()
            lr = 8e-3

        if not args.eval_only:
            train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel",
                  save=filename, optimtype=torch.optim.AdamW, lr=lr, weight_decay=0.01,
                  objective=torch.nn.BCEWithLogitsLoss())

        print(f"Testing {filename}")
        model = torch.load(filename).cuda()

        tmp = test(model, testdata, method_name=fusion_dict[args.fuse], dataset="imdb", criterion=torch.nn.BCEWithLogitsLoss(),
             task="multilabel", no_robust=True, measure_time=args.measure)

        log1.append(tmp['f1_micro'])
        log2.append(tmp['f1_macro'])

    print(log1, log2)
    print(f'Finish {args.n_runs} runs')
    print(f'f1 micro {np.mean(log1) * 100:.2f} ± {np.std(log1) * 100:.2f}')
    print(f'f1 macro {np.mean(log2) * 100:.2f} ± {np.std(log2) * 100:.2f}')
