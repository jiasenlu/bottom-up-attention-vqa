from __future__ import print_function
import os
import os.path as pth
import ujson as json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, coco_dir=None):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    coco_dir: if provided include paths to images
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    if coco_dir:
        coco_img_dir = pth.join(coco_dir, 'images', name + '2014')
        coco_img_format = pth.join(coco_img_dir, 'COCO_'+name+'2014_{:0>12d}.jpg').format

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))
        if coco_dir:
            entries[-1]['image_path'] = coco_img_format(img_id)
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', img_type='features',
                 coco_dir=None, img_size=256, crop_size=None, crop=None):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        # use pre-computed features or load images from file
        assert img_type in ['features', 'image']
        self.img_type = img_type
        self.img_size = img_size

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        if img_type == 'features':
            print('loading features from h5 file')
            h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
            with h5py.File(h5_path, 'r') as hf:
                self.features = np.array(hf.get('image_features'))
                self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, coco_dir)

        self.tokenize()
        self.tensorize()
        if img_type == 'features':
            self.v_dim = self.features.size(2)
            self.s_dim = self.spatials.size(2)
        if img_type == 'image':
            transform_lst = [
                transforms.Resize((img_size, img_size)),
            ]
            if crop == 'random':
                transform_lst.append(transforms.RandomCrop(crop_size))
            elif crop == 'center':
                transform_lst.append(transforms.CenterCrop(crop_size))
            transform_lst.extend([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            self.res_transform = transforms.Compose(transform_lst)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        if self.img_type == 'features':
            self.features = torch.from_numpy(self.features)
            self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def _spatial(self, features):
        # TODO: just a dummy; actually implement if needed
        if not hasattr(self, '_spatial_features'):
            self._spatial_features = torch.from_numpy(np.zeros(6))
        return self._spatial_features

    def __getitem__(self, index):
        entry = self.entries[index]
        if self.img_type == 'features':
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        elif self.img_type == 'image':
            img = default_loader(entry['image_path'])
            features = self.res_transform(img)
            spatials = self._spatial(features)

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        result = {
            'image': features,
            'spatial': spatials,
            'question': question,
            'answer': target,
        }
        return result

    def __len__(self):
        return len(self.entries)