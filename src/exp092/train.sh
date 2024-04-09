# python main.py --seed 2024 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 2025 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 2026 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 2027 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 2028 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir resnetrs50_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir resnetrs50_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir resnetrs50_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 3024 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 3025 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 3026 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 3027 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 3028 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir convnext_tiny_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir convnext_tiny_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir convnext_tiny_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 4024 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 4025 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 4026 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 4027 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 4028 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir caformer_s18_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir caformer_s18_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir caformer_s18_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 5024 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_224_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 5025 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_224_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 5026 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_224_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 5027 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_224_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 5028 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir swin_s3_small_224_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir swin_s3_small_224_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir swin_s3_small_224_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir swin_s3_small_224_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 6024 --model_name davit_small.msft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir davit_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 6025 --model_name davit_small.msft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir davit_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 6026 --model_name davit_small.msft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir davit_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 6027 --model_name davit_small.msft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir davit_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 6028 --model_name davit_small.msft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir davit_small_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir davit_small_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir davit_small_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir davit_small_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 7024 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 7025 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 7026 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 7027 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 7028 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir maxxvitv2_nano_rw_256_384_el30_mixup_50ep --batch_size 64 --num_workers 24


python main.py --seed 8024 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir tiny_vit_21m_384_el30_mixup_50ep --num_workers 6

python main.py --seed 8025 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir tiny_vit_21m_384_el30_mixup_50ep --num_workers 6

python main.py --seed 8026 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir tiny_vit_21m_384_el30_mixup_50ep --num_workers 6

python main.py --seed 8027 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir tiny_vit_21m_384_el30_mixup_50ep --num_workers 6

python main.py --seed 8028 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 96 --width 192 --logdir tiny_vit_21m_384_el30_mixup_50ep --num_workers 6
    
python eval.py --logdir tiny_vit_21m_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir tiny_vit_21m_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir tiny_vit_21m_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 9024 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir edgenext_base_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 9025 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir edgenext_base_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 9026 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir edgenext_base_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 9027 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir edgenext_base_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 9028 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir edgenext_base_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir edgenext_base_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir edgenext_base_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir edgenext_base_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 10024 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 10025 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 10026 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 10027 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 10028 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir inception_next_tiny_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir inception_next_tiny_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir inception_next_tiny_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 13024 --model_name caformer_m36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 12 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_m36_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 13025 --model_name caformer_m36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 12 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_m36_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 13026 --model_name caformer_m36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 12 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_m36_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 13027 --model_name caformer_m36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 12 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_m36_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 13028 --model_name caformer_m36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 12 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_m36_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir caformer_m36_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir caformer_m36_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir caformer_m36_384_el30_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 14024 --model_name inception_next_small.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 14025 --model_name inception_next_small.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 14026 --model_name inception_next_small.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 14027 --model_name inception_next_small.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_small_384_el30_mixup_50ep --num_workers 6

# python main.py --seed 14028 --model_name inception_next_small.sail_in1k --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_small_384_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir inception_next_small_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir inception_next_small_384_el30_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir inception_next_small_384_el30_mixup_50ep --batch_size 64 --num_workers 24

