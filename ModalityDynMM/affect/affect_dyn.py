import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP, Transformer, Sequential, Identity
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
    def __init__(self, temp, hard_gate, freeze=True, model_name_list=None):
        super(DynMMNet, self).__init__()
        self.branch_num = 3
        self.encoders, self.heads = self.load_model(model_name_list)
        if freeze:
            self.freeze_model()

        # gating network
        # self.gate = GRU(409, 3, dropout=True, has_padding=True)
        self.gate = nn.Sequential(Transformer(409, 10), nn.Linear(10, self.branch_num))
        self.temp = temp
        self.hard_gate = hard_gate
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = 0

    def load_model(self, model_name_list):
        encoder_list, head_list = [], []
        if model_name_list is not None:
            for model_name in model_name_list:
                encoder_list.append(torch.load(model_name))
                head_list.append(torch.load(model_name.replace('encoder', 'head')))
                print(f'Loading model {model_name}')
            return nn.ModuleList(encoder_list), nn.ModuleList(head_list)

    def freeze_model(self):
        for param in self.encoders.parameters():
            param.requires_grad = False
        for param in self.heads.parameters():
            param.requires_grad = False

    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        print(self.weight_list)
        tmp = torch.mean(self.weight_list, dim=0)
        print(f'mean branch weight {tmp[0].item():.4f}, {tmp[1].item():.4f}, {tmp[2].item():.4f}')
        self.store_weight = False

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()

    def forward2(self, inputs):
        x = torch.cat(inputs[0], dim=2)
        weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        pred_list = []
        for i in range(len(inputs[0])):
            mid = self.encoders[i]([inputs[0][i], inputs[1][i]])
            pred_list.append(self.heads[i](mid))

        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1]
        if self.infer_mode == -1:
            weight = torch.ones_like(weight) / self.branch_num

        output = weight[:, 0:1] * pred_list[0] + weight[:, 1:2] * pred_list[1] + weight[:, 2:3] * pred_list[2]
        return output, weight[:, 2].mean()

    def forward(self, inputs, path, weight_enable):
        if weight_enable:
            x = torch.cat(inputs[0], dim=2)
            weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        mid = self.encoders[path]([inputs[0][path], inputs[1][path]])
        output = self.heads[path](mid)
        return output


class DynMMNetV2(nn.Module):
    def __init__(self, temp, hard_gate, freeze, model_name_list):
        super(DynMMNetV2, self).__init__()
        self.branch_num = 2
        self.text_encoder = torch.load(model_name_list[0])
        self.text_head = torch.load(model_name_list[0].replace('encoder', 'head'))
        self.branch2 = torch.load(model_name_list[1])

        if freeze:
            self.freeze_branch(self.text_encoder)
            self.freeze_branch(self.text_head)
            self.freeze_branch(self.branch2)

        self.gate = nn.Sequential(Transformer(409, 10), nn.Linear(10, self.branch_num))
        self.temp = temp
        self.hard_gate = hard_gate
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = 0
        self.flop = torch.Tensor([135.13226, 320.03205])
        # self.flop = torch.Tensor([156.02, 340.92])

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
        # print('path 0', torch.where(self.weight_list[:, 0] == 1))
        # print('path 1', torch.where(self.weight_list[:, 1] == 1))
        return tmp[1].item()

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()

    def forward(self, inputs):
        x = torch.cat(inputs[0], dim=2)
        weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))

        pred_list = [self.text_head(self.text_encoder([inputs[0][2], inputs[1][2]])), self.branch2(inputs)]
        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1], 0
        if self.infer_mode == -1:
            weight = torch.ones_like(weight) / self.branch_num

        output = weight[:, 0:1] * pred_list[0] + weight[:, 1:2] * pred_list[1]
        return output, weight[:, 1].mean()

    def forward_separate_branch(self, inputs, path, weight_enable):  # see separate branch performance
        if weight_enable:
            x = torch.cat(inputs[0], dim=2)
            weight = DiffSoftmax(self.gate([x, inputs[1][0]]), tau=self.temp, hard=self.hard_gate)
        if path == 1:
            output = self.text_head(self.text_encoder([inputs[0][2], inputs[1][2]]))
        else:
            output = self.branch2(inputs)
        return output


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("unimodal network on mosi",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--data", type=str, default='mosei', help="dataset: mosi / mosei")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--enc", type=str, default='transformer', help="gru / transformer")
    argparser.add_argument("--n-epochs", type=int, default=50, help="number of epochs")
    argparser.add_argument("--temp", type=float, default=1, help="temperature")
    argparser.add_argument("--hard-gate", action='store_true', help='hard gates')
    argparser.add_argument("--reg", type=float, default=0.0, help="reg loss weight")
    argparser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
    argparser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    argparser.add_argument("--infer-mode", type=int, default=0, help="inference mode")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--freeze", action='store_true', help='freeze other parts of the model')
    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load data
    datafile = {'mosi': 'mosi_raw', 'mosei': 'mosei_senti_data'}
    traindata, validdata, testdata = get_dataloader('./data/' + datafile[args.data] + '.pkl', robust_test=False,
                                                    data_type=args.data, num_workers=16)

    log = np.zeros((args.n_runs, 5))
    for n in range(args.n_runs):
        # Init Model
        # modality = ['visual', 'audio', 'text']
        # model_name_list = []
        # for m in modality:
        #     model_name_list.append(os.path.join('./log', args.data, 'reg_' + args.enc + '_encoder_' + m + '.pt'))
        # model = DynMMNet(3, args.temp, args.hard_gate, args.freeze, model_name_list).cuda()
        model_name_list = ['./log/mosei/b1_reg_transformer_encoder_text.pt', './log/mosei/b2_lf_tran.pt']
        model = DynMMNetV2(args.temp, args.hard_gate, args.freeze, model_name_list).cuda()
        filename = os.path.join('./log', args.data, 'dyn_enc_' + args.enc + '_reg_' + str(args.reg) +
                                'freeze' + str(args.freeze) + '.pt')

        # Train
        if not args.eval_only:
            train(None, None, None, traindata, validdata, args.n_epochs, task="regression", optimtype=torch.optim.AdamW,
                  is_packed=True, early_stop=True, lr=args.lr, save=filename, weight_decay=args.wd,
                  objective=torch.nn.L1Loss(), moe_model=model, additional_loss=True, lossw=args.reg)

        # Test
        print(f"Testing model {filename}:")
        model = torch.load(filename).cuda()
        model.infer_mode = args.infer_mode
        print('-' * 30 + 'Val data' + '-' * 30)
        tmp = test(model=model, test_dataloaders_all=validdata, dataset=args.data, is_packed=True,
                   criterion=torch.nn.L1Loss(reduction='sum'), task='posneg-classification', no_robust=True, additional_loss=True)

        model.reset_weight()
        print('-' * 30 + 'Test data' + '-' * 30)
        tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data, is_packed=True,
                   criterion=torch.nn.L1Loss(reduction='sum'), task='posneg-classification', no_robust=True, additional_loss=True)
        log[n] = tmp['Accuracy'], tmp['Loss'], tmp['Corr'], model.cal_flop(), model.weight_stat()

    print(log[:, 0])
    print(log[:, 1])
    print(log[:, 2])
    print(log[:, 3])
    print(log[:, 4])
    print('-' * 60)
    print(f'Finish {args.n_runs} runs')
    print(f'Test Accuracy {np.mean(log[:, 0]) * 100:.2f} ± {np.std(log[:, 0]) * 100:.2f}')
    print(f'Loss {np.mean(log[:, 1]):.4f} ± {np.std(log[:, 1]):.4f}')
    print(f'Corr {np.mean(log[:, 2]):.4f} ± {np.std(log[:, 2]):.4f}')
    print(f'FLOP {np.mean(log[:, 3]):.2f} ± {np.std(log[:, 3]):.2f}')
    print(f'Ratio {np.mean(log[:, 4]):.3f} ± {np.std(log[:, 4]):.2f}')

    idx = np.argmax(log[:, 1])
    print('Best result', log[idx, :])