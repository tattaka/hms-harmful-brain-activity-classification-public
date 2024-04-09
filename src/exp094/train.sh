# python main.py --seed 2124 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_192_el30_mixup_100ep --num_workers 6

# python main.py --seed 2125 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_192_el30_mixup_100ep --num_workers 6

# python main.py --seed 2126 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_192_el30_mixup_100ep --num_workers 6

# python main.py --seed 2127 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_192_el30_mixup_100ep --num_workers 6

# python main.py --seed 2128 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 30 \
#     --lr 1e-3 --batch_size 16 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_2_5d_192_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir resnetrs50_2_5d_192_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir resnetrs50_2_5d_192_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir resnetrs50_2_5d_192_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 3124 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 3125 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 3126 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 3127 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 3128 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir convnext_tiny_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24


python main.py --seed 4124 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --logdir caformer_s18_2_5d_256_el30_mixup_100ep --num_workers 6

python main.py --seed 4125 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --logdir caformer_s18_2_5d_256_el30_mixup_100ep --num_workers 6

python main.py --seed 4126 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --logdir caformer_s18_2_5d_256_el30_mixup_100ep --num_workers 6

python main.py --seed 4127 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --logdir caformer_s18_2_5d_256_el30_mixup_100ep --num_workers 6

python main.py --seed 4128 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
    --logdir caformer_s18_2_5d_256_el30_mixup_100ep --num_workers 6
    
python eval.py --logdir caformer_s18_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir caformer_s18_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir caformer_s18_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 5124 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 112 --width 224 --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --num_workers 6

# python main.py --seed 5125 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 112 --width 224 --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --num_workers 6

# python main.py --seed 5126 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 112 --width 224 --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --num_workers 6

# python main.py --seed 5127 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 112 --width 224 --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --num_workers 6

# python main.py --seed 5128 --model_name swin_s3_small_224.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 112 --width 224 --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir swin_s3_small_2_5d_224_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 6124 --model_name davit_small.msft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir davit_small_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 6125 --model_name davit_small.msft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir davit_small_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 6126 --model_name davit_small.msft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir davit_small_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 6127 --model_name davit_small.msft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir davit_small_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 6128 --model_name davit_small.msft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir davit_small_2_5d_256_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir davit_small_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir davit_small_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir davit_small_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 7124 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 7125 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 7126 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 7127 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 7128 --model_name maxxvitv2_nano_rw_256.sw_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 2e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#      --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir maxxvitv2_nano_rw_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 8124 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 8125 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 8126 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 8127 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 8128 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# # python make_pseudo_label_wo_leak_train.py --logdir tiny_vit_21m_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 9124 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir edgenext_base_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 9125 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir edgenext_base_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 9126 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir edgenext_base_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 9127 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir edgenext_base_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 9128 --model_name edgenext_base.in21k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir edgenext_base_2_5d_256_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir edgenext_base_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir edgenext_base_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir edgenext_base_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 10124 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 10125 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 10126 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 10127 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 10128 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir inception_next_tiny_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24


# python main.py --seed 11124 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 11125 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 11126 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 11127 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --num_workers 6

# python main.py --seed 11128 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 100 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --num_workers 6
    
# python eval.py --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir swinv2_tiny_window16_2_5d_256_el30_mixup_100ep --batch_size 64 --num_workers 24



# python main.py --seed 12124 --model_name resnetrs101 --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 0 --gpus 4 --epochs 150 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir resnetrs101_2_5d_256_el45_mixup_100ep --num_workers 6

# python main.py --seed 12125 --model_name resnetrs101 --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 1 --gpus 4 --epochs 150 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir resnetrs101_2_5d_256_el45_mixup_100ep --num_workers 6

# python main.py --seed 12126 --model_name resnetrs101 --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 2 --gpus 4 --epochs 150 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir resnetrs101_2_5d_256_el45_mixup_100ep --num_workers 6

# python main.py --seed 12127 --model_name resnetrs101 --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 3 --gpus 4 --epochs 150 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir resnetrs101_2_5d_256_el45_mixup_100ep --num_workers 6

# python main.py --seed 12128 --model_name resnetrs101 --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 4 --gpus 4 --epochs 150 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --logdir resnetrs101_2_5d_256_el45_mixup_100ep --num_workers 6
    
# python eval.py --logdir resnetrs101_2_5d_256_el45_mixup_100ep --batch_size 64 --num_workers 24
