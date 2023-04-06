import argparse
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch

from datasets.affect.get_data import get_dataloader
from unimodals.common_models import GRU, MLP, Transformer, Sequential, Identity, GRUWithLinear
from fusions.mult import MULTModel
from fusions.common_fusions import ConcatEarly, Concat, LowRankTensorFusion
from training_structures.Supervised_Learning import train, test


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("unimodal network on mosi", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--data", type=str, default='mosei', help="dataset: mosi / mosei")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--fusion", type=int, default=3, help="0-4")
    argparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='measure inference time')
    args = argparser.parse_args()

    # args.eval_only = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load data
    datafile = {'mosi': 'mosi_raw', 'mosei': 'mosei_senti_data'}
    traindata, validdata, testdata = get_dataloader('./data/multibench/' + datafile[args.data] + '.pkl', robust_test=False,
                                                    data_type=args.data, num_workers=0)
    log = np.zeros((args.n_runs, 3))
    for n in range(args.n_runs):
        # Init Model
        if args.fusion == 0:
            encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
            head = Sequential(GRU(409, 512, dropout=True, has_padding=True), MLP(512, 256, 1)).cuda()
            fusion = ConcatEarly().cuda()

        elif args.fusion == 1:
            encoders = [GRU(35, 64, dropout=True, has_padding=True).cuda(),
                        GRU(74, 128, dropout=True, has_padding=True).cuda(),
                        GRU(300, 512, dropout=True, has_padding=True).cuda()]
            head = MLP(704, 512, 1).cuda()
            # encoders = [GRU(35, 70, dropout=True, has_padding=True).cuda(),
            #             GRU(74, 200, dropout=True, has_padding=True).cuda(),
            #             GRU(300, 600, dropout=True, has_padding=True).cuda()]
            # head = MLP(870, 870, 1).cuda()
            fusion = Concat().cuda()

        elif args.fusion == 2:
            encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
            head = Sequential(Transformer(409, 300).cuda(), MLP(300, 128, 1)).cuda()
            fusion = ConcatEarly().cuda()

        elif args.fusion == 3:
            encoders = [Transformer(35, 60).cuda(),
                        Transformer(74, 120).cuda(),
                        Transformer(300, 120).cuda()]
            head = MLP(300, 128, 1).cuda()
            fusion = Concat().cuda()

        elif args.fusion == 4:
            class HParams():
                num_heads = 10
                layers = 4
                attn_dropout = 0.1
                attn_dropout_modalities = [0, 0, 0.1]  # [0.2] * 1000
                relu_dropout = 0.1
                res_dropout = 0.1
                out_dropout = 0.1
                embed_dropout = 0.2
                embed_dim = 40
                attn_mask = True
                output_dim = 1
                all_steps = False


            encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
            fusion = MULTModel(3, [35, 74, 300], hyp_params=HParams).cuda()
            head = Identity().cuda()

        else:
            encoders = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).cuda(),
                        GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).cuda(),
                        GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).cuda()]
            head = MLP(128, 512, 1).cuda()
            fusion = LowRankTensorFusion([32, 32, 128], 128, 32).cuda()


        fusion_name = {0: 'ef_gru', 1: 'lf_gru', 2: 'ef_tran', 3: 'lf_tran', 4: 'mult', 5:'lrtf'}
        print(f'Fusion model {fusion_name[args.fusion]}')
        os.makedirs(os.path.join('./log', args.data), exist_ok=True)
        filename = os.path.join('./log', args.data, fusion_name[args.fusion] + '.pt')

        # Train
        if not args.eval_only:
            train(encoders, fusion, head, traindata, validdata, 1000, task="regression", optimtype=torch.optim.AdamW,
                  is_packed=True,early_stop=True, lr=args.lr, save=filename, weight_decay=args.wd,
                  objective=torch.nn.L1Loss())

        # Test
        print(f"Testing model {filename}:")
        model = torch.load(filename).cuda()
        print('Val data')
        tmp = test(model=model, test_dataloaders_all=validdata, dataset=args.data, is_packed=True,
             criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, measure_time=args.measure)

        print('Test data')
        tmp = test(model=model, test_dataloaders_all=testdata, dataset=args.data, is_packed=True,
             criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True, measure_time=args.measure)
        log[n] = tmp['Accuracy'], tmp['Loss'], tmp['Corr']

    print(log)
    print(f'Finish {args.n_runs} runs')
    print(f'Test Accuracy {np.mean(log[:, 0]) * 100:.2f} ± {np.std(log[:, 0]) * 100:.2f}')
    print(f'Loss {np.mean(log[:, 1]):.4f} ± {np.std(log[:, 1]):.4f}')
    print(f'Corr {np.mean(log[:, 2]):.4f} ± {np.std(log[:, 2]):.4f}')

    # Test data ent ent 0.5716876983642578
