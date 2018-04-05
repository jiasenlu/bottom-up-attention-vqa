import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import os.path as pth

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train',
                        choices=['train', 'eval'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--workers', type=int, default=10,
                        help='number of data loader workers')
    parser.add_argument('--coco_dir', type=str,
                        help='Directory of MS COCO (with images/)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--start_from', type=str,
                        help='path of weights to initialize vqa model with')
    parser.add_argument('--vision', type=str, choices=['bottomup', 'res101',
                        'res152'], help='visual features to use')
    args = parser.parse_args()
    return args


def load_model(dataset, args):
    constructor = 'build_%s' % args.model
    if args.vision in ['res101', 'res152']:
        # cnn_backed, cnns_dir
        cnn_args = (args.vision, 'data/cnns/')
    else:
        cnn_args = None
    model = getattr(base_model, constructor)(dataset, args.num_hid, cnn_args)
    if not args.start_from:
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model)
    if args.start_from:
        state_dict = torch.load(args.start_from)
        model.load_state_dict(state_dict)
    model = model.cuda()
    return model

def get_feature_map_size(model):
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torchvision.set_image_backend('accimage')

    # data
    batch_size = args.batch_size
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    img_type = 'features' if args.vision == 'bottomup' else 'image'
    eval_dset = VQAFeatureDataset('val', dictionary, img_type=img_type,
                                  coco_dir=args.coco_dir,
                                  img_size=args.image_size,
                                  crop_size=args.crop_size,
                                  crop='center' if args.crop_size else None)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True,
                             num_workers=args.workers, pin_memory=True)

    # model
    model = load_model(eval_dset, args)

    if args.task == 'train':
        model.train(True)
        train_dset = VQAFeatureDataset('train', dictionary, img_type=img_type,
                                       coco_dir=args.coco_dir,
                                       img_size=args.image_size,
                                       crop_size=args.crop_size,
                                       crop='random' if args.crop_size else None)
        train_loader = DataLoader(train_dset, batch_size, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)

        train(model, train_loader, eval_loader, args.epochs, args.output)

    elif args.task == 'eval':
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        eval_log = pth.join(args.output, 'results.txt')
        with open(eval_log, 'w') as f:
            f.write('eval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))