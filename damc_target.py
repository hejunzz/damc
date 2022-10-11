from __future__ import print_function
import argparse
from damc_functions import *

from torch import multiprocessing
from torch.utils.tensorboard import SummaryWriter


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
G_scale_dict = {"visda": 0.1,
                "office-31": 0.1,
                "office-home": 0.1}
lr_dict = { "visda": 0.01,
            "office-31": 0.01,
            "office-home": 0.01}
resnet_dict = {"visda": '101', "office-home": '50', "office-31": '50'}

# Training settings
def init_main():
    parser = argparse.ArgumentParser(description='DAMC Domain Adaptation')
    parser.add_argument('--gpuid', type=str, default='1', metavar='G',
                        help='which gpu device')
    parser.add_argument("--task", default='office-home', type=str)
    parser.add_argument("--source", default='Product')  # train
    parser.add_argument("--target", default='RW')  # validation #webcam
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  # visda 0.01, office-31 3e-3
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                        help='the name of optimizer')
    parser.add_argument('--seed', type=int, default=2021, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', type=str, default='save/icml-min005', metavar='B',
                        help='board dir')
    parser.add_argument('--resnet', type=str, default='101', metavar='B',
                        help='which resnet 18,50,101,152,200')
    parser.add_argument('--num_c', type=int, metavar='A', default=65,
                        help='number of sub-classifiers')
    parser.add_argument('--epoch_size', type=int, metavar='A', default=10000,
                        help='sample size of one epoch')
    parser.add_argument('--src_max_epoch', type=int, default=1, metavar='N',
                        help='number of epochs of overall train (default: 100)')
    parser.add_argument('--tgt_max_epoch', type=int, default=50, metavar='E',
                        help='the epoch that computes pseudo label')
    parser.add_argument('--model_ep', type=int, default=250, metavar='E',
                        help='load src model at ep')
    parser.add_argument('--p_start', type=int, default=1, metavar='N',
                        help='the epoch that begin using pseudo label')
    parser.add_argument('--pseudo_interval', type=int, default=2, metavar='N',
                        help='how many times to update pseudo labels')
    parser.add_argument('--pseudo_beta', type=float, default=0.1,
                        help='trade-off of pseudo label loss')
    parser.add_argument('--src_alpha', type=float, default=1,
                        help='load the source mode trained by source alpha hyper-parameter')
    parser.add_argument('--tgt_alpha', type=float, default=0.5,
                        help='target_alpha for minimizing pair of trace loss')
    parser.add_argument('--bn_dim', type=int, metavar='A', default=256,
                        help='bottleneck dimension')
    parser.add_argument('--smoothing', type=int, metavar='A', default=1,
                        help='smothing version')
    parser.add_argument('--epsilon', type=float, default=0.02,
                        help='label smoothing')
    val = 0
    args = parser.parse_args()
    args.num_classes = num_classes_dict[args.task]
    args.cls_middle = cls_middle_dict[args.task]
    args.num_layer = num_layer_dict[args.task]
    args.cls_prob = cls_dropout_dict[args.task]
    args.resnet = resnet_dict[args.task]
    args.lr = lr_dict[args.task]
    args.G_scale = G_scale_dict[args.task]
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # writer.add_hparams(hparam_dict=vars(args))

    return args


''' Begin  '''
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    mp_lock = multiprocessing.RLock()

    args = init_main()

    args.cuda = torch.cuda.is_available()
    source_path = task_path_dict[args.task] + args.source 
    target_path = task_path_dict[args.task] + args.target  
    # num_layer = args.num_layer
    batch_size = args.batch_size

    data_transforms_train = {
        source_path: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_path: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_transforms_val = {
        source_path: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_path: transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms_train[x]) for x in [source_path, target_path]}
    dsets_val = {x: datasets.ImageFolder(os.path.join(x), data_transforms_val[x]) for x in [source_path, target_path]}
    dset_sizes = {x: len(dsets[x]) for x in [source_path, target_path]}
    args.classes = dsets[source_path].classes
    print('classes' + str(args.classes))
    use_gpu = torch.cuda.is_available()

    src_train_len = int(len(dsets[source_path]) * 0.9)
    src_val_len = len(dsets[source_path]) - src_train_len
    print("Source %d: split into %d train and %d validation"%(len(dsets[source_path]), src_train_len, src_val_len))
    src_train_dataset, src_val_dataset = torch.utils.data.random_split(dsets[source_path], [src_train_len, src_val_len])
    src_train_loader = torch.utils.data.DataLoader(
            src_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)
    src_val_loader = torch.utils.data.DataLoader(
            src_val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

    tgt_dataset = IndexTensorDataset(dsets[target_path])
    tgt_loader = torch.utils.data.DataLoader(
            tgt_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)

    testset_loader = torch.utils.data.DataLoader(
            dsets_val[target_path],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)

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

    #opt_src = make_op_set(G, MC, lr, g_scale=0.1, bot_scale=1.0, src=True, args=args)
    opt_tgt = make_op_set(G, MC, lr, g_scale=0.1, bot_scale=0.1, c_scale=0.1, src=False, args=args)

    task_name = "TEST0-%s-%s-cls%d-lr=%.2e-pl=%.2f" % \
                (args.source, args.target, args.num_c,
                 opt_tgt['opt_g'].param_groups[0]['lr'],
                 args.pseudo_beta)
    args.log_dir = 'tf-%s/%s' % (args.task, task_name)
    writer = SummaryWriter(log_dir=args.log_dir)

    # source_model_pretrain(G, MC, opt_src, src_train_loader, src_val_loader, args, writer)
    alpha = int(args.src_alpha) if args.src_alpha >=1 else args.src_alpha
    args.src_alpha = alpha
    load_model(G, MC, args.model_ep, args=args)
    target_model_adaptation(G, MC, opt_tgt,src_train_loader, tgt_loader, pseudo_loader, testset_loader, args,writer)



