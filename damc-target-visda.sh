#!/bin/bash
echo "Begin Visda DA experiments"
num_c=12    # number of classifiers, for visda the optimal is just the number of categories due to sufficient synthetic source domain 
model_ep=10 # load pre-trained src model at ep
tgt_max_ep=30   # the maximum SF adaptation epochs
save="save/"    # model check point directory
gpu="0"         # use which gpu
task="visda"    # SFUDA task name
target="validation" # target domain, for visda the data is saved in "$damc_dir/visda/validation"
p_start=2       # the epoch that will use pseudo label
seed=2021       # random seed for reproducing experiment
interval=2      # the frequency of updating pseudo labels
bndim=256       # dimension of bottle neck layer
beta=0.01       # coefficient of pseudo label loss
alpha=0.3       # used to load the src mode trained by a particular src_alpha hyper-parameter
alphat=0.1      # coefficient of pair of trace loss
smo=0           # 0: doest not use label smoothing, should be consisitent with the pre-trained source model
epsilon=0.0     # label smoothing, valid if smoothig==1

python damc_target.py --tgt_alpha $alphat --epsilon $epsilon --p_start $p_start --save $save --smoothing $smo --model_ep $model_ep --tgt_max_epoch $tgt_max_ep --src_alpha $alpha --bn_dim $bndim --gpuid $gpu --task $task --pseudo_interval $interval --pseudo_beta $beta --seed $seed --num_c $num_c --target $target
