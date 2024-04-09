# python main.py --seed 2424 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 2425 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 2426 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 2427 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 2428 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_384_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir resnetrs50_2_5d_384_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 3424 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 3425 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 3426 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 3427 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_2_5d_384_el30_mixup_100ep --num_workers 6

# python main.py --seed 3428 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_2_5d_384_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir convnext_tiny_2_5d_384_el30_mixup_100ep --batch_size 64 --num_workers 24


python main.py --seed 4424 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir caformer_s18_2_5d_384_el40_mixup_100ep --num_workers 6

python main.py --seed 4425 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir caformer_s18_2_5d_384_el40_mixup_100ep --num_workers 6

python main.py --seed 4426 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir caformer_s18_2_5d_384_el40_mixup_100ep --num_workers 6

python main.py --seed 4427 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir caformer_s18_2_5d_384_el40_mixup_100ep --num_workers 6

python main.py --seed 4428 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir caformer_s18_2_5d_384_el40_mixup_100ep --num_workers 6
    
python eval.py --logdir caformer_s18_2_5d_384_el40_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 8424 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir tiny_vit_21m_2_5d_384_el40_mixup_100ep --num_workers 6

# python main.py --seed 8425 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir tiny_vit_21m_2_5d_384_el40_mixup_100ep --num_workers 6

# python main.py --seed 8426 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir tiny_vit_21m_2_5d_384_el40_mixup_100ep --num_workers 6

# python main.py --seed 8427 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir tiny_vit_21m_2_5d_384_el40_mixup_100ep --num_workers 6

# python main.py --seed 8428 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir tiny_vit_21m_2_5d_384_el40_mixup_100ep --num_workers 6
    
# python eval.py --logdir tiny_vit_21m_2_5d_384_el40_mixup_100ep --batch_size 32 --num_workers 24


# python main.py --seed 10424 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 10425 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 10426 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 10427 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 10428 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_2_5d_384_el20_mixup_100ep --num_workers 6
    
# python eval.py --logdir inception_next_tiny_2_5d_384_el20_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 5424 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 5425 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 5426 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 5427 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_2_5d_384_el20_mixup_100ep --num_workers 6

# python main.py --seed 5428 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 4 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_2_5d_384_el20_mixup_100ep --num_workers 6
    
# python eval.py --logdir swin_s3_small_2_5d_384_el20_mixup_100ep --batch_size 32 --num_workers 24

