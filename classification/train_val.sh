now=$(date +"%Y%m%d_%H%M%S")
nohup python -m torch.distributed.launch --nproc_per_node=4 train_imagenet.py \
--config configs/config_resnet50_2by4.yaml 2>&1|tee train-$now.log &





