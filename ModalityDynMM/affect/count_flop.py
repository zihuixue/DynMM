import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from thop import profile, clever_format

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import MLP, GRU, Transformer
from affect_dyn import DynMMNet, DynMMNetV2

def load_model(model_type, enc_type):
    um_dict = {0: 'visual', 1: 'audio', 2: 'text'}
    fusion_dict = {3: 'ef_gru', 4: 'lf_gru', 5: 'ef_tran', 6: 'lf_tran', 7: 'mult', 8: 'lrtf'}
    if model_type in fusion_dict:
        filename = os.path.join('./log', 'mosei', fusion_dict[model_type] + ".pt")
        model = torch.load(filename)
        print(f"Loading model {filename}")

    elif model_type in um_dict:
        enc_fp = os.path.join('./log', 'mosei', 'reg_' + enc_type + '_encoder_' + um_dict[model_type] + '.pt')
        encoder = torch.load(enc_fp)
        head = torch.load(enc_fp.replace('encoder', 'head'))
        model = nn.Sequential(encoder, head)
        print(f'Loading model {enc_fp}')

    else:
        # modality = ['visual', 'audio', 'text']
        # model_name_list = []
        # for m in modality:
        #     model_name_list.append(os.path.join('./log', 'mosei', 'reg_' + args.enc + '_encoder_' + m + '.pt'))
        # model = DynMMNet(3, 1, True, True, model_name_list)
        model_name_list = ['./log/mosei/reg_transformer_encoder_text.pt', './log/mosei/lf_tran.pt']
        model = DynMMNetV2(1, True, False, model_name_list)

    return model.cuda()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("imdb", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--weight-enable", action='store_true', help='whether to include gating networks')
    argparser.add_argument("--path", type=int, default=0, help="path: 0/1")
    argparser.add_argument("--enc", type=str, default='transformer', help="gru / transformer")
    argparser.add_argument("--model-type", type=int, default=-1, help="2-4: fusion, -1: dynamic")
    args = argparser.parse_args()

    model = load_model(args.model_type, args.enc)
    x = [[torch.randn(1, 50, 35).cuda(), torch.randn(1, 50, 74).cuda(), torch.randn(1, 50, 300).cuda()],
         [50 * torch.ones(1).long()] * 3]
    traindata, validdata, testdata = get_dataloader('./data/mosei_senti_data.pkl', robust_test=False, num_workers=0)
    if args.model_type < 0:
        macs, param = profile(model, inputs=(x, args.path, args.weight_enable), )
    elif args.model_type < 3:
        macs, param = profile(model, inputs=([x[0][args.model_type].float().cuda(), x[1][args.model_type]],))
    else:
        macs, param = profile(model, inputs=(x, ))
    macs, param = clever_format([macs, param], "%.5f")
    print(macs, param)