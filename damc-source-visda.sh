#!/bin/bsh
echo "Begin pre-training the source model on the Visda-2017 source domain"
num_c=12        # number of classifiers, for visda the optimal is just the number of categories due to sufficient synthetic source domain 
src_max_ep=20   # the maximum training epochs, the more the better, and we will take model selection among all the check-points
gpu="0"         # use which gpu
task="visda"    # SFUDA task name
save="save/"    # model check point directory
source="train"  # souce domain, for visda the data is saved in "$damc_dir/visda/train"
# target="validation"
seed=2021   # random seed for reproducing experiment
bndim=256   # dimension of bottle neck layer
alpha=0.3   # coefficient of adversarial discrepancy loss
smooth=0    # 0: doest not use label smoothing
epsilon=0   # epsilon of label smoothing when smooth==1
thresh=0.4  # threshold of worst case optimization, for visda we use 0.4
modelep=0   # 0: from scratch, does not resume from a check-point

python damc_source.py --model_ep $modelep --threshold $thresh --save $save --epsilon $epsilon --mul $mul --src_max_epoch $src_max_ep --smoothing $smooth --tgt_max_epoch $tgt_max_ep --src_alpha $alpha --bn_dim $bndim --gpuid $gpu --task $task --seed $seed --num_c $num_c --source $source