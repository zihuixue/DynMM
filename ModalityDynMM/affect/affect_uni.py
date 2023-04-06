import argparse
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.common_models import GRU, MLP, Transformer
from training_structures.unimodal import train, test
from datasets.affect.get_data import get_dataloader
import torch

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("unimodal network on mosi", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--mod", type=int, default=2, help="0/1/2")
    argparser.add_argument("--enc", type=str, default='transformer', help="encoder architecture: gru or transformer")
    argparser.add_argument("--hidden-dim1", type=int, default=0, help="hidden dimension 1")
    argparser.add_argument("--hidden-dim2", type=int, default=0, help="hidden dimension 1")
    argparser.add_argument("--data", type=str, default='mosei', help="dataset: mosi / mosei")
    argparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--clf", action='store_true', help='classification model, otherwise regression')
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='no training')
    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # args.eval_only = True

    # Load data
    # mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
    datafile = {'mosi': 'mosi_raw', 'mosei': 'mosei_senti_data'}
    traindata, validdata, testdata = get_dataloader('./data/multibench/'+datafile[args.data]+'.pkl', num_workers=0, robust_test=False, data_type=args.data)

    log = np.zeros((args.n_runs, 3))
    for n in range(args.n_runs):
        if args.mod == 0:
            input_dim = 35
            mod_name = 'visual'
            if args.enc == 'gru':
                hidden_dim1, hidden_dim2 = 64, 32
            else:
                hidden_dim1, hidden_dim2 = 120, 64

        elif args.mod == 1:
            input_dim = 74
            mod_name = 'audio'
            if args.enc == 'gru':
                hidden_dim1, hidden_dim2 = 128, 64
            else:
                hidden_dim1, hidden_dim2 = 120, 64

        else:
            input_dim = 300
            mod_name = 'text'
            if args.enc == 'gru':
                hidden_dim1, hidden_dim2 = 512, 256
            else:
                hidden_dim1, hidden_dim2 = 120, 64

        if args.hidden_dim1 > 0:
            hidden_dim1 = args.hidden_dim1
        if args.hidden_dim2 > 0:
            hidden_dim2 = args.hidden_dim2

        # mosi/mosei
        if args.enc == 'gru':
            encoder = GRU(input_dim, hidden_dim1, dropout=True, has_padding=True).cuda()
        else:
            encoder = Transformer(input_dim, hidden_dim1).cuda()
        output_dim = 2 if args.clf else 1
        head = MLP(hidden_dim1, hidden_dim2, output_dim).cuda()

        os.makedirs(os.path.join('./log', args.data), exist_ok=True)
        task_name = 'clf' if args.clf else 'reg'
        encoder_name = os.path.join('./log', args.data, task_name + '_' + args.enc + '_encoder_' + mod_name + '.pt')
        head_name = os.path.join('./log', args.data, task_name + '_' + args.enc + '_head_' + mod_name + '.pt')

        task = 'posneg-clf' if args.clf else 'regression'
        criterion = torch.nn.CrossEntropyLoss() if args.clf else torch.nn.L1Loss()
        print(f'unimodal training, modality {mod_name}, task {task}')
        if not args.eval_only:
            train(encoder, head, traindata, validdata, 100, task=task, optimtype=torch.optim.AdamW, lr=args.lr,
                  weight_decay=0.01, criterion=criterion, save_encoder=encoder_name, save_head=head_name,
                  early_stop=True, modalnum=args.mod, is_packed=True)

        # else:
        #     encoder_name = encoder_name.replace('reg', 'b1_reg')
        #     head_name = head_name.replace('reg', 'b1_reg')

        print(f"Testing model {encoder_name} | {head_name}:")
        encoder = torch.load(encoder_name).cuda()
        head = torch.load(head_name).cuda()
        print('Val data')
        tmp = test(encoder, head, validdata, 'affect', criterion=torch.nn.L1Loss(), task="posneg-classification",
             modalnum=args.mod, no_robust=True, is_packed=True, measure_time=args.measure)

        print('Test data')
        tmp = test(encoder, head, testdata, 'affect', criterion=torch.nn.L1Loss(), task="posneg-classification",
             modalnum=args.mod, no_robust=True, is_packed=True, measure_time=args.measure)
        log[n] = tmp['Accuracy'], tmp['Loss'], tmp['Corr']

    print(log)
    print(f'Finish {args.n_runs} runs')
    print(f'Test Accuracy {np.mean(log[:, 0]) * 100:.2f} ± {np.std(log[:, 0]) * 100:.2f}')
    print(f'Loss {np.mean(log[:, 1]):.4f} ± {np.std(log[:, 1]):.2f}')
    print(f'Corr {np.mean(log[:, 2]):.4f} ± {np.std(log[:, 2]):.2f}')

    # Test data ent: 0.5764016509056091