import numpy as np
import torch
# import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.autograd import Function
from torch.autograd import Variable

import time

import random
from itertools import combinations
from damc_helper import *


class IndexTensorDataset(Dataset):
    def __init__(self, dataset):
        super(IndexTensorDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index][0]
        target = self.dataset[index][1]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_layer=2, num_unit=256, prob=0.5, middle=1000):
        super(ResClassifier, self).__init__()
        middle2 = middle
        layers = []
        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer-1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle2))
            layers.append(nn.BatchNorm1d(middle2, affine=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle2, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x)
        return x        


class ResBase(nn.Module):
    def __init__(self, option='resnet18', bottleneck=512, pret=True, bot=True):
        super(ResBase, self).__init__()
        if option == 'resnet18':
            model_resnet = models.resnet18(pretrained=pret)
        if option == 'resnet50':
            model_resnet = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_resnet = models.resnet101(pretrained=pret)

        mod = list(model_resnet.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        self.in_features = model_resnet.fc.in_features
        self.bot = bot

        if bot:
            self.bottleneck_dim = bottleneck
            self.bottleneck = nn.Linear(self.in_features, self.bottleneck_dim)
            #self.bottleneck.apply(init_weights)
            nn.init.normal_(self.bottleneck.weight.data, 0, 0.005)
            nn.init.constant_(self.bottleneck.bias.data, 0.1)
            self.bot_bn = nn.BatchNorm1d(self.bottleneck_dim, affine=True)
        else:
            self.bottleneck_dim = self.in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.bot:
            x = self.bottleneck(x)
            x = self.bot_bn(x)
            x = F.relu(x)
            # dropout
            x = F.dropout(x, training=self.training)
        return x


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


def ent(output):
    # return - torch.sum(output * torch.log(output + 1e-6))
    return - torch.mean(output * torch.log(output + 1e-6))


"""
Discrepancy used for source model pretraining to push away each pair of classifiers
"""
def discrepancy(out1, out2, detach=False):
    out1_d = out1.clone().detach()
    out2_d = out2.clone().detach()
    return torch.mean(torch.sum(torch.abs(out1_d - out2_d), dim=1))\
        if detach else torch.mean(torch.sum(torch.abs(out1 - out2), dim=1))


def trace_loss(out):
    """
        Representation of agreement of many-classifiers
            0: agreement of all classifiers  and all output is one-hot
            1: disagreement of any pair of classifiers
    """
    prod = out[0] * 1
    for i in range(1, len(out)):
        prod *= out[i]
    tr = torch.sum(prod) / prod.shape[0]

    return 1 - tr


def pair_trace_loss(out, max=True):
    num_c = len(out)
    combs = list(combinations(range(num_c), 2))
    dist = []
    n = 0
    for p in combs:
        n += 1
        trloss = trace_loss([out[p[0]], out[p[1]]])
        dist.append(trloss)

    dist_tensor = torch.stack(dist)
    return torch.max(dist_tensor) if max else torch.mean(dist_tensor)


"""
Select the minimum close pair of classifiers for our worst case optimization
"""
def min_simplex_discrepancy(out):
    num_c = len(out)
    combs = list(combinations(range(num_c), 2))
    dist = []
    n = 0
    for p in combs:
        n += 1
        dist.append(discrepancy(out[p[0]], out[p[1]], detach=False))

    dist_tensor = torch.stack(dist)
    return torch.min(dist_tensor), combs[torch.argmin(dist_tensor)], dist_tensor


"""
Only used for checking worst case optimization
"""
def min_k_discrepancy(out, k):
    num_c = len(out)
    combs = list(combinations(range(num_c), 2))
    dist = []
    n = 0
    for p in combs:
        n += 1
        dist.append(trace_loss([out[p[0]], out[p[1]]]))
        # dist.append(discrepancy(out[p[0]], out[p[1]], detach=True))

    dist_tensor = torch.stack(dist)
    sorted_tensor, sorted_idx = torch.sort(dist_tensor)
    cls_pairs = [combs[sorted_idx[i]] for i in range(k)]
    return dist_tensor, sorted_tensor[:k], cls_pairs


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def make_op_set(G, C, lr, g_scale, bot_scale, c_scale, args, src=True):
    param_group = []
    opt_set = {}
    for k, v in G.named_parameters():
        if 'bot' not in k:
            param_group += [{'params': v, 'lr': lr * g_scale}]  # for visda 0.01, for office 0.05 + 3e-3---》91.2
    opt_g = optim.SGD(param_group)
    opt_g = op_copy(opt_g)
    opt_set['opt_g'] = opt_g

    param_group = []
    for k, v in G.named_parameters():
        if 'bot' in k:
            param_group += [{'params': v, 'lr': lr * bot_scale}]
    opt_bn = optim.SGD(param_group)
    opt_bn = op_copy(opt_bn)
    opt_set['opt_bn'] = opt_bn

    if src:
        param_group = []
        for i in range(args.num_c):
            for k, v in C[i].named_parameters():
                param_group += [{'params': v, 'lr': lr * c_scale}]
        opt_c = optim.SGD(param_group)
        opt_c = op_copy(opt_c)
        opt_set['opt_c'] = opt_c

    return opt_set


def reset_grad(opt):
    opt['opt_g'].zero_grad()
    if 'opt_c' in opt: 
        opt['opt_c'].zero_grad()
    opt['opt_bn'].zero_grad()


def damc_test(G, MC, epoch, testset_loader, args, max_sample=55388, target=True):
    G.eval()
    for C in MC:
        C.eval()

    test_loss = 0
    test_loss_ent = 0
    corrects = np.zeros(args.num_c)
    correct_ens = 0
    size = 0

    size_dict = dict.fromkeys(range(args.num_classes), 0)
    correct_dict = dict.fromkeys(range(args.num_classes), 0)
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes)

    correct_dict_list = []
    for i in range(len(MC)):
        correct_dict_list.append(dict.fromkeys(range(args.num_classes), 0))

    with torch.no_grad():
        for batch_idx, data in enumerate(testset_loader):
            if args.max_sample > 0 and batch_idx * args.batch_size > args.max_sample:
                break
            img, label = data
            if args.cuda:
                img, label = img.cuda(), label.cuda()
            # img, label = Variable(img, volatile=True), Variable(label)

            feat = G(img)

            # NEED improve!!!!
            output_ensemble = 0
            # output_ensemble_s = 0
            prob_set = []
            for i, C in enumerate(MC):
                output = C(feat)
                p = F.softmax(output, dim=1)
                prob_set.append(p)
                output_ensemble += output
                pred = output.data.max(1)[1]
                corrects[i] += pred.eq(label.data).cpu().sum()
                for j in range(label.data.size()[0]):
                    if (pred[j].item() == label[j].item()):
                        correct_dict_list[i][pred[j].item()] = correct_dict_list[i][pred[j].item()] + 1

            pred = output_ensemble.data.max(1)[1]

            for t, p in zip(label.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            for l in label:
                size_dict[l.item()] = size_dict[l.item()] + 1

            for i in range(label.data.size()[0]):
                if (pred[i].item() == label[i].item()):
                    correct_dict[pred[i].item()] = correct_dict[pred[i].item()] + 1

            correct_ens += pred.eq(label.data).cpu().sum()

            test_loss += F.nll_loss(output_ensemble, label).data
            k = label.data.size()[0]
            size += k

    test_loss = test_loss / size
    test_loss_ent = test_loss_ent / size
    tm = time.strftime('%x %X ')
    cls_str = ""
    cls_acc = np.zeros(args.num_c)
    for i in range(args.num_c):
        cls_str += 'cls-%d: %.3f; ' % (i, 100. * corrects[i] / size)
        cls_acc[i] = 100. * corrects[i] / size
    cls_str += "\n Mean acc: %.4f, std: %.4f, max-min: %.4f\n" % (np.mean(cls_acc), np.std(cls_acc),
                                                                  np.max(cls_acc) - np.min(cls_acc))

    if args.task == 'visda':  # or args.task =='office-home':
        per_class_acc_str = ''
        per_class_acc_list = []
        for i in range(len(correct_dict)):
            per_class_acc_list.append(100. * correct_dict[i] / size_dict[i])
            per_class_acc_str += 'cls-label-%s: %d/%d = %.3f\n' % (
            args.classes[i], correct_dict[i], size_dict[i], per_class_acc_list[i])
        cls_str += per_class_acc_str
        cls_str += 'ep%d-per-class acc: %.3f\n' % (epoch, np.mean(per_class_acc_list))
    cls_str += 'ep%d-cls-ens-T: %d/%d = %.3f\n' % (epoch, correct_ens, size, 100. * (float(correct_ens) / size))

    if not target:
        cls_category_matrix = np.zeros((args.num_c, args.num_classes))
        pretty_print_cls = min(len(args.classes), len(args.classes))
        cls_detail = 'cls-no ' + '\t'.join(args.classes[:pretty_print_cls]) + '\n'
        for k in range(len(MC)):
            per_class_acc_str_list = []
            per_class_acc_list = []
            for i in range(len(correct_dict)):
                per_class_acc_str_list.append('%6.2f'%(100. * correct_dict_list[k][i] / size_dict[i]))
                per_class_acc_list.append(100. * correct_dict_list[k][i] / size_dict[i])
                cls_category_matrix[k,i] =  100. * correct_dict_list[k][i] / size_dict[i]
                #per_class_acc_str += 'cls-label-%s: %d/%d = %.3f\n' % (
                #    args.classes[i], correct_dict_list[k][i], size_dict[i], per_class_acc_list[i])
            cls_detail += 'cls-%2d: ' % k + '  '.join(per_class_acc_str_list[:pretty_print_cls]) + '\n'
            #cls_str += 'cls-%d: \n' % k
            #cls_str += per_class_acc_str
            cls_str += 'cls%2d-ep%d-per-class acc: %.3f\n' % (k, epoch, np.mean(per_class_acc_list))
        str_voting = 'category voting: '
        voting_list = []
        for i in range(len(correct_dict)):
            str_voting += '%6.2f  ' % (np.sum(cls_category_matrix[:, i] > 99.99) / args.num_c)
            voting_list.append(np.sum(cls_category_matrix[:, i] > 99.99) / args.num_c)
        if args.num_classes < 15:
            cls_str += cls_detail
        cls_str += str_voting
        voting = np.asarray(voting_list)
        cls_str += '\n sum of voting: >50%%=%d, 0%%=%d, 100%%=%d' % (sum(voting>=0.5), sum(voting==0), sum(voting==1))

    msg = tm + 'Epoch:{} Loss: {:.4f}，Loss_ensemble: {:.4f} '.format(epoch, test_loss, test_loss_ent) + cls_str
    print(msg)

    # if epoch > 20 and epoch % 5 == 0:
    # if epoch >= 0 and epoch % 1 == 0:
    #     tsne_visualize(epoch=epoch, n_batch=32)
    #     fig = plot_confusion_matrix(confusion_matrix.numpy(), args.classes, epoch, args)
    #     writer.add_figure('Confusion matrix', fig, epoch)

    # if target:
    #     writer.add_scalar('%s-%s: Acc' % (args.source, args.target), float(correct_ens) / size, epoch)
    #     writer.add_scalar('%s-%s: Acc class' % (args.source, args.target), np.mean(per_class_acc_list), epoch)
    #     writer.add_text('%s-%s: Per-class accuracy summary' % (args.source, args.target), msg, epoch)

    return 100 * (float(correct_ens) / size), confusion_matrix.cpu().numpy()


def damc_source_model_pretrain(G, MC, opt_s, src_loader, val_loader, args, writer=None):
    criterion = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=args.epsilon).cuda() if args.smoothing else nn.CrossEntropyLoss().cuda()
    max_iter = args.src_max_epoch * max(len(src_loader), 5000 / args.batch_size)  # (args.epoch_size // batch_size)  # 20 * (30000//batch_size)
    # max_iter = 10 * max(len(src_loader), 5000 / args.batch_size)  # (args.epoch_size // batch_size)  # 20 * (30000//batch_size)
    iter_num_1 = args.model_ep * len(src_loader)

    for ep in range(1, args.src_max_epoch+1):
        G.train()
        for C in MC:
            C.train()

        for batch_idx, data in enumerate(src_loader):
            iter_num_1 += 1
            if args.max_sample > 0 and batch_idx * args.batch_size > args.max_sample:
                break

            data, target = data
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # when pretraining network source only
            target = Variable(target)
            data = Variable(data)

            lr_scheduler(opt_s['opt_c'], iter_num=iter_num_1, max_iter=max_iter)
            lr_scheduler(opt_s['opt_bn'], iter_num=iter_num_1, max_iter=max_iter)
            lr_scheduler(opt_s['opt_g'], iter_num=iter_num_1, max_iter=max_iter)

            reset_grad(opt_s)
            feat_s = G(data)
            loss_xent = 0
            for C in MC:
                c_out = C(feat_s)
                loss_xent += criterion(c_out, target)

            loss_xent.backward()
            G_norm = check_gradient_norm(G)
            gradient_scaling(G.parameters(), args.num_c)
            G_norm_1 = check_gradient_norm(G)
            opt_s['opt_g'].step()
            opt_s['opt_bn'].step()
            opt_s['opt_c'].step()
            loss_adv = 0

            # adversarial training in source domain
            if args.src_alpha > 0:
                reset_grad(opt_s)
                with torch.no_grad():
                    feat_s = G(data)

                adv_list = []
                pair_list = []
                adv_i = 0
                loss_adv = 0
                while loss_adv < args.threshold and adv_i < 6: 
                    #reset_grad(opt_s)
                    adv_i += 1
                    output_s = []
                    for C in MC:
                        c_out = C(feat_s)
                        p = F.softmax(c_out, dim=1)
                        output_s.append(p)
                    loss_adv, c_pair, _ = min_simplex_discrepancy(output_s)
                    loss = -args.src_alpha * loss_adv

                    adv_list.append(loss_adv)
                    pair_list.append(c_pair)

                    loss.backward()
                    # commented lines produce good result for visda alpha=0.4 2021-12-14
                    C_norm = check_gradient_norm(MC[c_pair[0]])
                    opt_s['opt_c'].step()

            if batch_idx % args.log_interval == 0:
                msg = 'Ep %d-%d: source xent %.4f, adv-C %.4f, G lr %.2e, |G1| %.4f,  |C| %.4f\n' % \
                      (ep, batch_idx, loss_xent, loss_adv, opt_s['opt_g'].param_groups[0]['lr'],
                       G_norm_1, C_norm)

                k_str = ['{:d}-{:d}: {:.4f}'.format(pair_list[k][0], pair_list[k][1], adv_list[k])
                         for k in range(len(pair_list))]
                msg += '\tpairs {:d}: '.format(adv_i) + ', '.join(k_str)
                print(msg)

        # validating the source model here
        tm = time.strftime('%x %X ')
        if args.task == 'visda':
            acc = damc_test(G, MC, epoch=ep, testset_loader=val_loader, args=args, target=True)
            acc = damc_test(G, MC, epoch=ep, testset_loader=src_loader, args=args, target=False)
            msg = source_model_selection(G, MC, val_loader, args)
            print(tm + 'Ep %d: classifiers discrepancy %s' % (ep, msg))
            if ep + args.model_ep > 1:
                save_model(G, MC, ep + args.model_ep, args)
        else:
            if ep % 10 == 0:
                acc = damc_test(G, MC, epoch=ep, testset_loader=val_loader, args=args, target=False)
                #msg = source_model_selection(G, MC, val_loader, args)
                print(tm + 'Ep %d: classifiers discrepancy %s' % (ep, msg))
                if ep >= 150 and ep % 10==0:
                    #acc = tesstt(G, MC, epoch=ep, testset_loader=src_loader, args=args, target=False)
                    save_model(G, MC, ep + args.model_ep, args)


def source_model_selection(G, MC, val_loader, args):
    G.eval()
    for C in MC:
        C.eval()

    # total_adv = 0
    with torch.no_grad():
        output = []
        first_flag = True
        for batch_idx, data in enumerate(val_loader):
            if batch_idx * args.batch_size > 10000:
                break

            data, target = data
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # when pretraining network source only
            # target = Variable(target)
            data = Variable(data)

            feat = G(data)
            # loss_adv = 0

            for i, C in enumerate(MC):
                c_out = C(feat)
                p = F.softmax(c_out, dim=1)
                if not first_flag:
                    output[i] = torch.cat((output[i], p), dim=0)
                else:
                    output.append(p)
            first_flag = False

        loss_pair = - pair_trace_loss(output)  # args.num_c/2 * simplex_discrepancy(output_s, args.mul)
        loss_tr = trace_loss(output)
        # dis_tensor, min_k_dis, min_k_pair = min_k_discrepancy(output_s, args.num_c)
        # min_dis, c_pair, dis_tensor = min_simplex_discrepancy(output)
        msg = 'Trace loss: {:.4f}, Pair trace: {:.4f}\n'.format(loss_tr, loss_pair)
        #k_str = ['{:d}-{:d}: {:.4f}'.format(min_k_pair[k][0], min_k_pair[k][1], min_k_dis[k])
        #         for k in range(len(min_k_pair))]
        #msg += ', '.join(k_str)

    return msg


def damc_target_model_adaptation(G, MC, opt_t, tgt_loader, pseudo_loader, testset_loader, args, writer=None):
    criterion = nn.CrossEntropyLoss().cuda()
    max_iter = args.tgt_max_epoch * max(len(tgt_loader), 5000 / args.batch_size)  # (args.epoch_size // batch_size)  # 20 * (30000//batch_size)
    iter_num_2 = 0
    p_start = args.p_start

    for C in MC:
        C.eval()

    for ep in range(1, args.tgt_max_epoch+1):
        if args.pseudo_interval != 0:
            if ep == p_start or ep > p_start and (ep-p_start) % args.pseudo_interval == 0:   # when ep=1 or every interval update pseudo label
                G.eval()
                pseudo_label = obtain_pseudo_label(pseudo_loader, G, MC,ep%args.num_c,args)
                pseudo_label = torch.from_numpy(pseudo_label).cuda()

        G.train()
        for batch_idx, data in enumerate(tgt_loader):
            iter_num_2 += 1

            if args.max_sample > 0 and batch_idx * args.batch_size > args.max_sample:
                break

            data, target, idx_target = data
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # when pretraining network source only
            target = Variable(target)
            data = Variable(data)
            
            lr_scheduler(opt_t['opt_bn'], iter_num=iter_num_2, max_iter=max_iter)
            lr_scheduler(opt_t['opt_g'], iter_num=iter_num_2, max_iter=max_iter)

            reset_grad(opt_t)
            feat_t = G(data)
            loss_pseudo = 0
            if args.pseudo_interval != 0 and ep >= p_start:
                pseudo_tgt = pseudo_label[idx_target]
            output_t = []
            ent_loss = 0
            mean_ent_loss = 0
            for i, C in enumerate(MC):
                c_out = C(feat_t)
                if args.pseudo_interval != 0 and ep >= p_start: # and i == 0:
                    # only consider pseudo label loss for the first classifier
                    loss_pseudo += criterion(c_out, pseudo_tgt)
                p = F.softmax(c_out, dim=1)
                output_t.append(p)
                pm = torch.mean(p, 0)
                ent_loss += ent(p)
                mean_ent_loss += ent(pm)

            mean_ent_loss /= args.num_c
            ent_loss /= args.num_c
            loss_pseudo /= args.num_c
            loss_tr = pair_trace_loss(output_t, max=False) if args.tgt_alpha > 0 else 0 # simplex_discrepancy(output_t)  #
            loss_t = args.tgt_alpha * loss_tr + args.pseudo_beta * loss_pseudo - 0.1 * mean_ent_loss + 0.1 * ent_loss

            loss_t.backward()
            G_norm = check_gradient_norm(G)
            #gradient_scaling(G.parameters(), args.num_c)
            G_norm_c = check_gradient_norm(G)
            opt_t['opt_g'].step()
            opt_t['opt_bn'].step()
            with torch.no_grad():
                dis_tensor, min_k_dis, min_k_pair = min_k_discrepancy(output_t, 1)

            if batch_idx % args.log_interval == 0:
                tm = time.strftime('%x %X ')
                msg = tm + 'Train Epoch: {}/{}\t pseudo xent {:.4f},  trace loss{:.2e}, lr {:.2e}, '.format(
                   ep, batch_idx, loss_pseudo, loss_tr, opt_t['opt_g'].param_groups[0]['lr'])  # test 右边数第一个为loss
                # msg += 'ent: {:.4f}, marginal-ent: {:.4f}'.format(ent_loss, mean_ent_loss)
                msg += '\n\tmean:{:.4f}, std:{:.4f}, '.format(torch.mean(dis_tensor), torch.std(dis_tensor))
                k_str = [' {:d}-{:d}: {:.4f}'.format(min_k_pair[k][0], min_k_pair[k][1], min_k_dis[k])
                        for k in range(len(min_k_pair))]
                msg += ', '.join(k_str)
                msg += '|grad|: G-{:.4f} - Gs-{:.4f}'.format(G_norm, G_norm_c)
                print(msg)

                if writer:
                    writer.add_scalar('%s-%s: Trace Loss' % (args.source, args.target), loss_tr, iter_num_2)
                    writer.add_scalar('%s-%s: Pseudo Loss' % (args.source, args.target), loss_pseudo, iter_num_2)

        # end of one epoch training
        # if args.task == 'visda':
        damc_test(G=G, MC=MC, epoch=ep, testset_loader=testset_loader, args=args)        

