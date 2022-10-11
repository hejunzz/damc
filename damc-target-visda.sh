#!/bin/bash
echo "Begin Visda DA experiments"
num_c=12
model_ep=10
tgt_max_ep=30
save="save/icml-min04c1-s2023"
gpu="0"
exp_name="0123"
task="visda"
source="train"
target="validation"
p_start=2
seed=2023
interval=2
bndim=256
beta=0.01
alpha=0.3
epsilon=0.0
alphat=0.1
smo=0
# train --> validation
file=${exp_name}"-target"${interval}"-"${task}"-cls"${num_c}"-src"${model_ep}"-tgt"${tgt_max_ep}"-alpha"${alpha}$"-pseudo"$beta".out"
echo $file

#python damc_target.py --save $save --smoothing $smo --model_ep $model_ep --tgt_max_epoch $tgt_max_ep --src_alpha $alpha --bn_dim $bndim --gpuid $gpu --task $task --pseudo_interval $interval --pseudo_beta $beta --seed $seed --num_c $num_c --source $source --target $target 

python damc_target.py --tgt_alpha $alphat --epsilon $epsilon --p_start $p_start --save $save --smoothing $smo --model_ep $model_ep --tgt_max_epoch $tgt_max_ep --src_alpha $alpha --bn_dim $bndim --gpuid $gpu --task $task --pseudo_interval $interval --pseudo_beta $beta --seed $seed --num_c $num_c --source $source --target $target
