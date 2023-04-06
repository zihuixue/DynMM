import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import MLP, Linear, MaxOut_MLP
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test, MMDL


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class DynMMNet(nn.Module):
    def __init__(self, branch_num=2, pretrain=True, freeze=True):
        super(DynMMNet, self).__init__()
        self.branch_num = branch_num
        # branch 1: text network
        self.text_encoder = torch.load('./log/imdb/encoder_text.pt') if pretrain else MLP(300, 512, 512)
        self.text_head = torch.load('./log/imdb/head_text.pt') if pretrain else MLP(512, 512, 23)
        # self.branch1 = nn.Sequential(self.text_encoder, self.text_head)

        # branch2: image network, discard this branch due to poor performance
        self.image_encoder = torch.load('./log/imdb/encoder_image.pt') if pretrain else MLP(4096, 1024, 512)
        self.image_head = torch.load('./log/imdb/head_image.pt') if pretrain else MLP(512, 512, 23)
        # self.branch2 = nn.Sequential(self.image_encoder, self.image_head)

        # branch3: text+image late fusion
        if pretrain:
            self.branch3 = torch.load('./log/imdb/best_lf.pt')
        else:
            encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
            head = Linear(1024, 23)
            fusion = Concat()
            self.branch3 = MMDL(encoders, fusion, head, has_padding=False)

        if freeze:
            self.freeze_branch(self.text_encoder)
            self.freeze_branch(self.text_head)
            self.freeze_branch(self.image_encoder)
            self.freeze_branch(self.image_head)
            self.freeze_branch(self.branch3)

        # gating network
        self.gate = MLP(4396, 128, branch_num)
        self.temp = 1
        self.hard_gate = True
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = 0
        self.flop = torch.Tensor([1.25261, 10.86908])

    def freeze_branch(self, m):
        for param in m.parameters():
            param.requires_grad = False

    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        print(self.weight_list)
        tmp = torch.mean(self.weight_list, dim=0)
        print(f'mean branch weight {tmp[0].item():.4f}, {tmp[1].item():.4f}')
        self.store_weight = False
        return tmp[1].item()

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()

    def forward(self, inputs):
        x = torch.cat(inputs, dim=1)
        weight = DiffSoftmax(self.gate(x), tau=self.temp, hard=self.hard_gate)

        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        pred_list = [self.text_head(self.text_encoder(inputs[0])), self.branch3(inputs)]
        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1], 0

        output = weight[:, 0:1] * pred_list[0] + weight[:, 1:2] * pred_list[1]
        return output, weight[:, 1].mean()

    def forward_separate_branch(self, inputs, path, weight_enable):  # see separate branch performance
        if weight_enable:
            x = torch.cat(inputs, dim=1)
            weight = DiffSoftmax(self.gate(x), tau=self.temp, hard=self.hard_gate)
        if path == 1:
            output = self.text_head(self.text_encoder(inputs[0]))
        elif path == 2:
            output = self.image_head(self.image_encoder(inputs[1]))
        else:
            output = self.branch3(inputs)

        return output


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("imdb",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--data", type=str, default='imdb', help="dataset name")
    argparser.add_argument("--n-epochs", type=int, default=50, help="number of epochs")
    argparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    argparser.add_argument("--reg", type=float, default=0.1, help="reg loss weight")
    argparser.add_argument("--freeze", action='store_true', help='freeze branch weights')
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--hard", action='store_true', help='hard labels')
    argparser.add_argument("--no-pretrain", action='store_true', help='train from scratch')
    argparser.add_argument("--infer-mode", type=int, default=0, help="infer mode")
    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    traindata, validdata, testdata = get_dataloader("./data/multimodal_imdb.hdf5", "./data/mmimdb", vgg=True, batch_size=128, no_robust=True)

    log1, log2 = np.zeros((args.n_runs, 1)), np.zeros((args.n_runs, 3))
    for n in range(args.n_runs):
        # Init Model
        model = DynMMNet(pretrain=1-args.no_pretrain, freeze=args.freeze)
        filename = os.path.join('./log', args.data, 'DynMMNet_freeze' + str(args.freeze) + '_reg_' + str(args.reg) + '.pt')

        if not args.eval_only:
            model.hard_gate = args.hard
            train(None, None, None, traindata, validdata, args.n_epochs, task="multilabel", optimtype=torch.optim.AdamW,
                  is_packed=False, early_stop=True, lr=args.lr, save=filename, weight_decay=args.wd,
                  objective=torch.nn.BCEWithLogitsLoss(), moe_model=model, additional_loss=True, lossw=args.reg)

        # Test
        print(f"Testing model {filename}:")
        model = torch.load(filename).cuda()
        model.hard_gate = True

        # print('-' * 30 + 'Val data' + '-' * 30)
        model.infer_mode = args.infer_mode
        # tmp = test(model=model, test_dataloaders_all=validdata, dataset=args.data, is_packed=False,
        #            criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True, additional_loss=True)

        print('-' * 30 + 'Test data' + '-' * 30)
        model.reset_weight()
        tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data, is_packed=False,
                   criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True, additional_loss=True)
        log1[n] = model.weight_stat()
        log2[n] = tmp['f1_micro'], tmp['f1_macro'], model.cal_flop()

    print(log1)
    print(log2)
    print('-' * 60)
    print(f'Finish {args.n_runs} runs')
    # print(f'Val f1 micro {np.mean(log1[:, 0]) * 100:.2f} ± {np.std(log1[:, 0]) * 100:.2f} | f1 macro {np.mean(log1[:, 1]) * 100:.2f} ± {np.std(log1[:, 0]) * 100:.2f}')
    print(f'Test f1 micro {np.mean(log2[:, 0]) * 100:.2f} ± {np.std(log2[:, 0]) * 100:.2f} | '
          f'f1 macro {np.mean(log2[:, 1]) * 100:.2f} ± {np.std(log2[:, 0]) * 100:.2f} | ' 
          f'Flop saving {np.mean(log2[:, 2]):.2f} ± {np.std(log2[:, 2]):.2f}M | '
          f'Branch selection ratio {np.mean(log1):.3f} ± {np.std(log1):.3f}')
    idx = np.argmax(log2[:, 1])
    print('Best result', log2[idx, :])