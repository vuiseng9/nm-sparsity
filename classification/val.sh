now=$(date +"%Y%m%d_%H%M%S")
python  -m torch.distributed.launch --nproc_per_node=1 train_imagenet.py \
--config configs/config_resnet50_2by4.yaml \
--model_dir ../zoo \
--resume_from resnet50\(2_4\).pth \
--evaluate 2>&1|tee val-$now.log




