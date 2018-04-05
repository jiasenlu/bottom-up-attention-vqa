import torch
import torch.nn as nn
from torch.autograd import Variable
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, cnn=None):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.cnn = cnn

    def forward(self, v, b, q, labels, return_att=False):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        if self.cnn:
            with torch.no_grad():
                # relies on implementation with, e.g., resnet.cnn_net
                v = self.cnn.cnn_net(v)
            v = Variable(v.data)
            # pixels are objects
            v = v.view(v.shape[0], v.shape[1], -1).transpose(1, 2)

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        if return_att:
            return logits, att
        else:
            return logits


def load_cnn(cnn_backend, cnns_dir):
    from resnet import resnet
    if cnn_backend == 'res101':
        cnn = resnet(cnns_dir, _num_layers=101, pretrained=True)
    elif cnn_backend == 'res152':
        cnn = resnet(cnns_dir, _num_layers=152, pretrained=True)
    else:
        raise Exception('unknown cnn {}'.format(cnn_backend))
    cnn.requires_grad = False
    return cnn


def build_baseline0(dataset, num_hid, cnn_args=None):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    cnn = load_cnn(*cnn_args) if cnn_args else None
    v_dim = cnn.v_dim if cnn else dataset.v_dim
    v_att = Attention(v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    model = BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, cnn=cnn)
    return model


def build_baseline0_newatt(dataset, num_hid, cnn_args=None):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    cnn = load_cnn(*cnn_args) if cnn_args else None
    v_dim = cnn.v_dim if cnn else dataset.v_dim
    v_att = NewAttention(v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    model = BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, cnn=cnn)
    return model