from __future__ import print_function
import argparse
from damc_functions import *

from torch import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torchsampler import ImbalancedDatasetSampler
import matplotlib.pyplot as plt
import seaborn as sns


task_path_dict = {"visda": "~/jhe/visda/",
                  "office-home": "~/jhe/office-home/OfficeHomeDataset/",
                  "office-31": "~/jhe/office-31/office31/"}
num_classes_dict = {"visda": 12,
                    "office-home": 65,
                    "office-31": 31}
num_layer_dict = {"visda": 2,
                  "office-home": 1,
                  "office-31": 1}
cls_middle_dict = {"visda": 512, # 512
                   "office-home": 1024,
                   "office-31": 1024} #1024
cls_dropout_dict = {"visda": 0.5,
                  "office-home": 0.5,
                   "office-31": 0.5}
G_scale_dict = {"visda": 0.01, # 0.01
                "office-31": 0.01,
                "office-home": 0.01}
lr_dict = { "visda": 0.01,
            "office-31": 0.01,
            "office-home": 0.01}
resnet_dict = {"visda": '101', "office-home": '50', "office-31": '50'}

# Training settings
def init_main():
    parser = argparse.ArgumentParser(description='DAMC Domain Adaptation')
    parser.add_argument('--gpuid', type=str, default='0', metavar='G',
                        help='which gpu device')
    parser.add_argument("--task", default='office-home', type=str)
    parser.add_argument("--source", default='Art')  # train
    parser.add_argument("--target", default='Clipart')  # validation
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # writer.add_hparams(hparam_dict = {x: datasets.ImageFolder(os.path.join(x), data_transforms_train[x]) for x in [source_path, target_path]}
    dsets_val = {x: datasets.ImageFolder(os.path.join(x), data_transforms_val[x]) for x in [source_path, target_path]}
    dset_sizes = {x: len(dsets[x]) for x in [source_path, target_path]}
    args.classes = dsets[source_path].classes
    print('classes' + str(args.classes))
    use_gpu = torch.cuda.is_available()

    if args.task == 'visda': # or args.task == 'office-home':
        src_train_len = int(len(dsets[source_path]) * 0.9)
        src_val_len = len(dsets[source_path]) - src_train_len
        print("Source %d: split into %d train and %d validation" % (len(dsets[source_path]), src_train_len, src_val_len))
        src_train_dataset, src_val_dataset = torch.utils.data.random_split(dsets[source_path], [src_train_len, src_val_len])
    else:
        src_train_dataset = dsets[source_path]
        src_val_dataset = dsets_val[source_path]

    src_train_loader = torch.utils.data.DataLoader(
        src_train_dataset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(src_train_dataset),
        # shuffle=True,
        num_workers=4,
        drop_last=True)
    src_val_loader = torch.utils.data.DataLoader(
        src_val_dataset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(src_val_dataset),
        # shuffle=True,
        num_workers=4,
        drop_last=True)

    tgt_dataset = IndexTensorDataset(dsets[target_path])
    tgt_loader = torch.utils.data.DataLoader(
        tgt_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    testset_loader = torch.utils.data.DataLoader(
        dsets[target_path],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    ds_pseudo = IndexTensorDataset(dsets_val[target_path])
    pseudo_loader = torch.utils.data.DataLoader(
        ds_pseudo,
        batch_size=64,
        shuffle=False,
        num_workers=4)

    option = 'resnet' + args.resnet
    G = ResBase(option, bottleneck=args.bn_dim, bot=1)
    MC = []
    for i in range(args.num_c):
        if args.task == "visda":
            MC.append(ResClassifier(num_layer=args.num_layer, num_unit=G.bottleneck_dim,
                                    middle=args.cls_middle, num_classes=args.num_classes,
                                    prob=args.cls_prob))  #
        else:
            MC.append(ResClassifier(num_layer=args.num_layer, num_unit=G.bottleneck_dim,
                                    middle=args.cls_middle, num_classes=args.num_classes))
            # C.append(ResClassifier2(num_unit=G.bottleneck_dim,
            #                        middle=1024, num_classes=args.num_classes
            #                        ))  #

    for i in range(args.num_c):
        MC[i].apply(weights_init)

    lr = args.lr
    if args.cuda:
        G.cuda()
        for i in range(args.num_c):
            MC[i].cuda()

    opt_src = make_op_set(G, MC, lr, g_scale=args.g_scale, bot_scale=1, c_scale=1, src=True, args=args)

    task_name = "TEST0-%s-%s-cls%d-lr=%.2e-pl=%.2f" % \
                (args.source, args.target, args.num_c,
                 opt_src['opt_g'].param_groups[0]['lr'],
                 args.pseudo_beta)
    args.log_dir = 'tf-%s/%s' % (args.task, task_name)
    alpha = int(args.src_alpha) if args.src_alpha >=1 else args.src_alpha
    args.src_alpha = alpha
    writer = SummaryWriter(log_dir=args.log_dir)

    if args.model_ep > 0:
        load_model(G, MC, args.model_ep, args=args)

    # args.src_alpha = 0.2
    # fig, ax = plt.subplots()
    # labels = [label for _, label in src_train_loader.dataset.imgs]
    # classe_labels, counts = np.unique(labels, return_counts=True)
    # ax.bar(classe_labels, counts)
    # ax.set_xticks(classe_labels)
    # plt.show()

    source_model_pretrain(G, MC, opt_src, src_train_loader, src_val_loader, args, writer)
    # target_model_adaptation(G, C, opt_tgt, tgt_loader, pseudo_loader, testset_loader, args, writer)


