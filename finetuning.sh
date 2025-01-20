# 单卡训练
# Set the environment variable for CUDA
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --config_file config/cfg_odvg.py \
    --datasets config/data_dog_cat.json \
    --output_dir fine_tuning_output/output_dog_cat \
    --pretrain_model_path MODEL/groundingdino_swinb_cogcoor.pth \
    --options text_encoder_type="bert-base-uncased" \
    --save_results \
    --save_log

# 分布式训练
# GPU_NUM = 2
# python -m torch.distributed.launch  --nproc_per_node=$GPU_NUM main.py \
#         --output_dir fine_tuning_output/output_dog_cat \
#         -c config/cfg_odvg.py \
#         --datasets config/data_dog_cat.json  \
#         --pretrain_model_path /path/to/groundingdino_swint_ogc.pth \
#         --options text_encoder_type=/path/to/bert-base-uncased