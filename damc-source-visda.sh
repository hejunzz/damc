#!/bin/bsh
echo "Begin Visda DA experiments"
num_c=12
src_max_ep=20
tgt_max_ep=20
gpu="1"
task="visda"
save="save/icml-min04c1-s2021"
source="train"
target="validation"
seed=2021
interval=0
bndim=256
beta=0.01
alpha=0.3
mul=1
smooth=0
epsilon=0
thresh=0.4
modelep=0
# train --> validation
file="source"${interval}"-"${task}"-cls"${num_c}"-src"${src_max_ep}"-smooth"${smooth}"-mul"${mul}"-alpha"${alpha}".out"
echo $file

python damc_source.py --model_ep $modelep --threshold $thresh --save $save --epsilon $epsilon --mul $mul --src_max_epoch $src_max_ep --smoothing $smooth --tgt_max_epoch $tgt_max_ep --src_alpha $alpha --bn_dim $bndim --gpuid $gpu --task $task --seed $seed --num_c $num_c --source $source --target $target


#python damc_source.py --mul $mul --src_max_epoch $src_max_ep --smoothing $smooth --src_alpha $alpha --bn_dim $bndim --gpuid $gpu --task $task --pseudo_interval $interval --pseudo_beta $beta --seed $seed --num_c $num_c --source $source --target $target
