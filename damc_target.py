from __future__ import print_function
import argparse
from damc_core import *
import os

from torch import multiprocessing
# from torch.utils.tensorboard import SummaryWriter


# Training settings
def init_main():
    parser = argparse.ArgumentParser(description='DAMC Domain Adaptation')
    parser.add_argument('--gpuid', type=str, default='0', metavar='G',
                        help='which gpu device')
    parser.add_argument("--task", default='visda', type=str)
    parser.add_argument("--source", default='train')  # train
    parser.add_argument("--target", default='validation')  # validation
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',  # visda 0.01, office-31 3e-3
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                        help='the name of optimizer')
    parser.add_argument('--seed', type=int, default=2021, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', type=str, default='save/', metavar='B',
                        help='model check point dir')
    parser.add_argument('--resnet', type=str, default='101', metavar='B',
                        help='which resnet 18,50,101,152,200')
    parser.add_argument('--num_c', type=int, metavar='A', default=12,
                        help='number of classifiers')
    parser.add_argument('--max_sample', type=int, metavar='A', default=-1,
                        help='sample size for debugging only, should be -1 to use the whole dataset')
    parser.add_argument('--tgt_max_epoch', type=int, default=30, metavar='E',
                        help='the epoch that computes pseudo label')
    parser.add_argument('--model_ep', type=int, default=10, metavar='E',
                        help='load pre-trained src model at ep')
    parser.add_argument('--p_start', type=int, default=2, metavar='N',
                        help='the epoch that begin using pseudo label')
    parser.add_argument('--pseudo_interval', type=int, default=2, metavar='N',
                        help='the frequency of updating pseudo labels')
    parser.add_argument('--pseudo_beta', type=float, default=0.01,
                        help='coefficient of pseudo label loss')
    parser.add_argument('--src_alpha', type=float, default=0.3,
                        help='used to load the src mode trained by a particular src_alpha hyper-parameter')
    parser.add_argument('--tgt_alpha', type=float, default=0.1,
                        help='coefficient of pair of trace loss')
    parser.add_argument('--bn_dim', type=int, metavar='A', default=256,
                        help='bottleneck dimension')
    parser.add_argument('--smoothing', type=int, metavar='A', default=0,
                        help='smoothing version of classifier to increase generality, 0: do not use smoothe')
    parser.add_argument('--epsilon', type=float, default=0,
                        help='label smoothing, valid if smoothig==1')
    parser.add_argument('--cls_middle', type=int, default=512,  # for visda, middle layer is 512
                        help='The middle layer of classifier')
    parser.add_argument('--num_layer', type=int, default=2, # for visda, we use 256->512->512->12 as one classifier
                        help='The layers of classifier')                                                
    val = 0
    args = parser.parse_args()
    args.cls_prob = 0.5
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
    target_path = os.path.join(os.getcwd(), args.task, args.target)
    print(target_path)

    data_transforms_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    data_transforms_val = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
    dsets = datasets.ImageFolder(os.path.join(target_path), data_transforms_train)
    dsets_val = datasets.ImageFolder(os.path.join(target_path), data_transforms_val)
    args.classes = dsets.classes
    args.num_classes = len(args.classes)
    print('classes' + str(args.classes))

    tgt_dataset = IndexTensorDataset(dsets)
    tgt_loader = torch.utils.data.DataLoader(
            tgt_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True)

    testset_loader = torch.utils.data.DataLoader(
        dsets_val,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True)

    ds_pseudo = IndexTensorDataset(dsets_val)
    pseudo_loader = torch.utils.data.DataLoader(
        ds_pseudo,
        batch_size=64,
        shuffle=False,
        num_workers=2)

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

    for i in range(args.num_c):
        MC[i].apply(weights_init)

    lr = args.lr
    if args.cuda:
        G.cuda()
        for i in range(args.num_c):
            MC[i].cuda()

    opt_tgt = make_op_set(G, MC, lr, g_scale=0.1, bot_scale=0.1, c_scale=0.1, src=False, args=args)

    # args.log_dir = 'tf-%s/%s' % (args.task, task_name)
    # writer = SummaryWriter(log_dir=args.log_dir)
    if args.src_alpha >=1:
        args.src_alpha = int(args.src_alpha)
    load_model(G, MC, args.model_ep, args=args)
    damc_target_model_adaptation(G, MC, opt_tgt, tgt_loader, pseudo_loader, testset_loader, args,writer=None)