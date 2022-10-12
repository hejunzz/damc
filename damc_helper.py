import numpy as np
import io
import os
import time
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



def save_model(G, MC, epoch, args):
    checkpoint = {
        "G": G.state_dict()
    }
    for i, C in enumerate(MC):
        checkpoint["C" + str(i)] = C.state_dict()
    torch.save(checkpoint, os.path.join(args.save, 'damc.' + args.task + '.' +
                                        args.source + '.' + #args.target + '.' +
                                        'cls' + str(args.num_c) + '.' +
                                        'smo' + str(args.smoothing) + '.' +
                                        str(args.src_alpha) + '.' + str(epoch) + ".pth"))


def load_model(G, MC, epoch, args):
    model_path = os.path.join(args.save, 'damc.' + args.task + '.' +
                              args.source + '.' + # args.target + '.' +
                              'cls' + str(args.num_c) + '.' +
                              'smo' + str(args.smoothing) + '.' +
                              str(args.src_alpha) + '.' + str(epoch) + ".pth")
    print("Load model from " + model_path)
    checkpoint = torch.load(model_path) if args.cuda else \
                    torch.load(model_path, map_location=torch.device('cpu'))
    G.load_state_dict(checkpoint["G"])
    for i, C in enumerate(MC):
        C.load_state_dict(checkpoint["C" + str(i)])


def gradient_scaling(parameters, scale):
    parameters = [p for p in parameters if p.grad is not None]
    scale_coef = 1 / scale
    for p in parameters:
        p.grad.detach().mul_(scale_coef)


def check_gradient_norm(G):
    total_norm = 0
    for k, v in G.named_parameters():
        if 'bot' not in k:
            param_norm = v.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    G_norm = total_norm ** (1. / 2)

    return G_norm


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 5e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def plot_embedding(data, label, title, args):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    custom = [Line2D([],[],marker='o',linestyle='None'),
    Line2D([],[],marker='+',linestyle='None')]

    fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    plt.clf()
    sns.scatterplot(
        x=data[:, 0], y=data[:, 1],
        hue=label, 
        palette=sns.color_palette("hls", args.num_classes),
        marker="o",
        s=80,
        legend="full",
        alpha=0.3
    )

    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # plt.savefig(args.save+'/'+title+'.eps', format='eps')
    # return buf, fig


def tsne_embedding(G, MC, dataloader, args, epoch, n_batch=32):
    G.eval()
    for i in range(args.num_c):
        MC[i].eval()
    # plot t-sne
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    feat_list = []
    label_list = []

    for batch_idx, data in enumerate(dataloader):
        # print(batch_idx)
        if batch_idx > n_batch:
            break
        data, target = data
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            feat = G(data)
            feat_list.append(feat.detach().cpu().numpy())
            label_list.append(target.numpy())

    feat_np = np.vstack(feat_list)
    label_np = np.concatenate(label_list).reshape(-1)
    result = tsne.fit_transform(feat_np)
    return result, label_np


def tsne_visualize(G, MC, src_loader, tgt_loader, args, epoch=0, n_batch=16):
    G.eval()
    for i in range(args.num_c):
        MC[i].eval()
    # plot t-sne
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    feat_list_t = []
    feat_list_s = []
    label_list_t = []
    label_list_s = []
    for batch_idx, data in enumerate(src_loader):
        # print(batch_idx)
        if batch_idx > n_batch:
            break
        data, target = data
        data = data.cuda()
        with torch.no_grad():
            feat_s = G(data)
            feat_list_s.append(feat_s.detach().cpu().numpy())
            label_list_s.append(target.numpy())

    for batch_idx, data in enumerate(tgt_loader):
        # print(batch_idx)
        if batch_idx > n_batch:
            break
        data, target = data
        data = data.cuda()
        with torch.no_grad():
            feat_t = G(data)
            feat_list_t.append(feat_t.detach().cpu().numpy())
            label_list_t.append(target.numpy())

    t0 = time.time()
    feat_np_t = np.vstack(feat_list_t)
    feat_np_s = np.vstack(feat_list_s)
    label_np_t = np.concatenate(label_list_t).reshape(-1)
    label_np_s = np.concatenate(label_list_s).reshape(-1)
    label_np = np.concatenate([label_np_s, label_np_t]).reshape(-1)

    feat_np = np.vstack([feat_np_s, feat_np_t])

    result = tsne.fit_transform(feat_np)

    buf, fig = plot_embedding(result, label_np,
                    '%s-%s-%s-%d' % (args.task, args.source, args.target, epoch), args)

    # writer.add_figure(task_name, fig)
    #pyplot.show(fig)
    # import PIL.Image
    # image = PIL.Image.open(buf)
    # image = np.array(image)
    # writer.add_image('T-SNE', image, epoch, dataformats="HWC")



def obtain_pseudo_label(loader, netF, netC, cls_n, args):
    start_test = True
    t_g = 0
    t_mc = 0
    iSel = np.random.permutation(len(netC))[cls_n:cls_n+1]
    with torch.no_grad():
        iter_test = iter(loader)
        temmm = len(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            batch_idx = data[2]
            # label_extract = loader.dataset.dataset[1][batch_idx]
            # assert (sum(label_extract == labels) == len(labels))
            if args.cuda:
                inputs = inputs.cuda()  # .permute(0, 3, 1, 2)
            t0 = time.time()
            feas = netF(inputs)
            t_g = time.time() - t0
            outputs = 0
            t0 = time.time()
            #outputs = netC[iSel](feas)
            for i in iSel:
               outputs += netC[i](feas)
            #outputs avg
            outputs = outputs / len(iSel)
            t_mc = time.time() - t0
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
  
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.num_classes)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > 0)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset],'cosine')
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)


    log_str = ' Accuracy = {:.2f}% -> {:.2f}%, G: {:.2e} sec, MC: {:.2e} sec'.format(accuracy * 100, acc * 100, t_g,
                                                                                     t_mc)
    print(log_str + '\n')
    return predict.astype('int')