import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable

from itertools import islice

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    opt_parameters = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adamax(opt_parameters)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        total_disc_loss = 0
        train_score = 0
        t = time.time()

        for i, batch in enumerate(train_loader):
            v = Variable(batch['image']).cuda()
            b = Variable(batch['spatial']).cuda()
            q = Variable(batch['question']).cuda()
            a = Variable(batch['answer']).cuda()

            # VQA model (attention generator)
            pred, att = model(v, b, q, a, return_att=True)
            answer_loss = instance_bce_with_logits(pred, a)
            #att_loss = ...
            loss = answer_loss #+ att_loss
            loss.backward()
            nn.utils.clip_grad_norm(opt_parameters, 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        max_eval_batches = 100

        eval_score, bound = evaluate(model, eval_loader,
                             max_batches=max_eval_batches)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        #if eval_score > best_eval_score:
        model_path = os.path.join(output, 'model_ep{}.pth'.format(epoch))
        print('checkpointing model to {}...'.format(model_path))
        torch.save(model.state_dict(), model_path)
        print('done')
        #    best_eval_score = eval_score


def evaluate(model, dataloader, max_batches=-1):
    score = 0
    upper_bound = 0
    num_data = 0
    num_disc_data = 0
    batches = iter(dataloader)
    if max_batches > -1:
        batches = islice(batches, max_batches)
    for batch in batches:
        v = Variable(batch['image']).cuda()
        b = Variable(batch['spatial']).cuda()
        q = Variable(batch['question']).cuda()
        a = batch['answer']
        with torch.no_grad():
            # vqa
            pred, att = model(v, b, q, None, return_att=True)
            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)


    score = score / num_data
    upper_bound = upper_bound / num_data

    return score, upper_bound