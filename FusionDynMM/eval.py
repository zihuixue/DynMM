# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
import numpy as np
import random
import codecs
import csv
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.confusion_matrix import ConfusionMatrixTensorflow, ConfusionMatrixPytorch, miou_pytorch
from src.prepare_data import prepare_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_log(filename, args, data_mean, data_std):
    file = codecs.open(filename, 'a')
    writer = csv.writer(file)
    writer.writerow(
        [args.mode, args.noise, args.num_runs, data_mean, data_std])
    writer.writerow('')
    file.close()


if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Evaluation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', required=True, type=str, help='Path to the checkpoint of the trained model.')
    parser.add_argument('--hard', action="store_true", help='use hard gates during inference time')
    parser.add_argument('--mode', type=int, default=-1, help='mode -1: no noise, mode 0: rgb noise, mode 1: depth noise, mode2: rgb/depth noise')
    parser.add_argument('--num-runs', type=int, default=1, help='noise level')
    parser.add_argument('--noise', type=float, help='noise level')
    parser.add_argument('--ini', action="store_true", help='ini stage: no dependency between layers')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    _, data_loader, *add_data_loader = prepare_data(args, with_input_orig=True)
    if args.valid_full_res:
        # cityscapes only -> use dataloader that returns full resolution images
        data_loader = add_data_loader[0]

    n_classes = data_loader.dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.start_weight()
    model.hard_gate = args.hard
    model.ini_stage = args.ini
    model.baseline = args.baseline
    model.to(device)

    n_samples = 0

    confusion_matrices = dict()

    cameras = data_loader.dataset.cameras
    camera = cameras[0]   # nyudepth only have one camera
    result = np.zeros(args.num_runs, dtype=float)
    for r in range(args.num_runs):
        set_seed(r)
        confusion_matrices[camera] = dict()
        confusion_matrices[camera] = ConfusionMatrixPytorch(n_classes)
        miou = miou_pytorch(confusion_matrices[camera])
        # confusion_matrices[camera] = ConfusionMatrixTensorflow(n_classes)
        n_samples_total = len(data_loader.dataset)
        with data_loader.dataset.filter_camera(camera):
            for i, sample in enumerate(data_loader):
                n_samples += sample['image'].shape[0]
                print(f'\r{n_samples}/{n_samples_total}', end='')
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)  # (bs, 1, 480, 640)
                rand_val = random.random()
                if args.mode == 0:
                    if rand_val < 0.33:
                        image = image + args.noise * abs(image).mean() * torch.randn_like(image)
                elif args.mode == 1:
                    if rand_val < 0.33:
                        depth = depth + args.noise * abs(depth).mean() * torch.randn_like(depth)
                elif args.mode == 2:
                    if rand_val < 0.33:
                        image = image + args.noise * abs(image).mean() * torch.randn_like(image)
                    elif rand_val < 0.66:
                        depth = depth + args.noise * abs(depth).mean() * torch.randn_like(depth)

                label_orig = sample['label_orig']
                _, image_h, image_w = label_orig.shape

                with torch.no_grad():
                    if args.modality == 'rgbd':
                        inputs = (image, depth, True)
                    elif args.modality == 'rgb':
                        inputs = (image,)
                    elif args.modality == 'depth':
                        inputs = (depth,)

                    pred = model(*inputs)

                    pred = F.interpolate(pred, (image_h, image_w),
                                         mode='bilinear',
                                         align_corners=False)
                    pred = torch.argmax(pred, dim=1)

                    # ignore void pixels
                    mask = label_orig > 0
                    label = torch.masked_select(label_orig, mask)
                    pred = torch.masked_select(pred, mask.to(device))

                    # In the label 0 is void but in the prediction 0 is wall.
                    # In order for the label and prediction indices to match
                    # we need to subtract 1 of the label.
                    label -= 1

                    # copy the prediction to cpu as tensorflow's confusion
                    # matrix is faster on cpu
                    pred = pred.cpu()

                    label = label.numpy()
                    pred = pred.numpy()

                    # print('label shape, pred shape', label.shape, pred.shape)
                    # confusion_matrices[camera].update_conf_matrix(label, pred)
                    confusion_matrices[camera].update(torch.from_numpy(label), torch.from_numpy(pred))

                print(f'\r{i + 1}/{len(data_loader)}', end='')

        # miou, _ = confusion_matrices[camera].compute_miou()
        miou_val = miou.compute().data.numpy() * 100
        print(f'Run {r}, mIoU: {miou_val:0.2f}')
        result[r] = miou_val

    print(result)
    print(f'Mean {np.mean(result):.2f}, Std {np.std(result):.2f}')

    # confusion_matrices['all'] = ConfusionMatrixPytorch(n_classes)
    # confusion_matrices['all'] = ConfusionMatrixTensorflow(n_classes)

    # sum confusion matrices of all cameras
    # for camera in cameras:
    #     confusion_matrices['all'].overall_confusion_matrix = confusion_matrices[camera].overall_confusion_matrix
    # miou, _ = confusion_matrices['all'].compute_miou()
    # print(f'All Cameras, mIoU: {100*miou:0.2f}')

    # model.end_weight(print_each=True, print_flop=True)
