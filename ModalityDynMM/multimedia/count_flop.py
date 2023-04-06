import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from thop import profile, clever_format

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from imdb_dyn import DynMMNet


def load_model(model_type):
    um_dict = {0: 'text', 1: 'image'}
    fusion_dict = {2: 'ef', 3: 'lf', 4: 'lrtf', 5: 'mim'}
    if model_type in fusion_dict:
        filename = "./log/imdb/best_" + fusion_dict[model_type] + ".pt"
        model = torch.load(filename)
        print(f"Loading model {filename}")

    elif model_type in um_dict:
        encoder = torch.load('./log/imdb/encoder_' + um_dict[model_type] + '.pt')
        head = torch.load('./log/imdb/head_' + um_dict[model_type] + '.pt')
        model = nn.Sequential(encoder, head)
    else:
        model = DynMMNet(pretrain=True, freeze=False)

    return model.cuda()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("imdb", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--weight-enable", action='store_true', help='whether to include gating networks')
    argparser.add_argument("--path", type=int, default=0, help="path: 0/1")
    argparser.add_argument("--model-type", type=int, default=-1, help="2-4: fusion, -1: dynamic")
    args = argparser.parse_args()

    model = load_model(args.model_type)
    x = [torch.randn(1, 300).cuda(), torch.randn(1, 4096).cuda()]
    if args.model_type < 0:
        macs, param = profile(model, inputs=(x, args.path, args.weight_enable), )
    elif args.model_type < 2:
        macs, param = profile(model, inputs=(x[args.model_type],))
    else:
        macs, param = profile(model, inputs=(x, ))
    macs, param = clever_format([macs, param], "%.5f")
    print(macs, param)
