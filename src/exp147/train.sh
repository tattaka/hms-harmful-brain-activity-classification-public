# python main.py --seed 2224 --model_name resnetrs50 --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 2225 --model_name resnetrs50 --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 2226 --model_name resnetrs50 --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 2227 --model_name resnetrs50 --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 2228 --model_name resnetrs50 --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_768_el20_mixup_50ep --num_workers 6
    
# python eval.py --logdir resnetrs50_768_el20_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 3224 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 3225 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 3226 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 3227 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_768_el20_mixup_50ep --num_workers 6

# python main.py --seed 3228 --model_name convnext_tiny.fb_in1k --drop_path_rate 0.3 --eeg_length 20 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir convnext_tiny_768_el20_mixup_50ep --num_workers 6
    
# python eval.py --logdir convnext_tiny_768_el20_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 4224 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_768_el30_mixup_50ep --num_workers 6

# python main.py --seed 4225 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_768_el30_mixup_50ep --num_workers 6

# python main.py --seed 4226 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_768_el30_mixup_50ep --num_workers 6

# python main.py --seed 4227 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_768_el30_mixup_50ep --num_workers 6

# python main.py --seed 4228 --model_name caformer_s18.sail_in1k_384 --drop_path_rate 0.3 --eeg_length 30 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir caformer_s18_768_el30_mixup_50ep --num_workers 6
    
# python eval.py --logdir caformer_s18_768_el30_mixup_50ep --batch_size 64 --num_workers 24


python main.py --seed 8224 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 64 --width 128 --logdir tiny_vit_21m_512_el30_mixup_50ep --num_workers 6

python main.py --seed 8225 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 64 --width 128 --logdir tiny_vit_21m_512_el30_mixup_50ep --num_workers 6

python main.py --seed 8226 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 64 --width 128 --logdir tiny_vit_21m_512_el30_mixup_50ep --num_workers 6

python main.py --seed 8227 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 64 --width 128 --logdir tiny_vit_21m_512_el30_mixup_50ep --num_workers 6

python main.py --seed 8228 --model_name tiny_vit_21m_384.dist_in22k_ft_in1k --drop_path_rate 0.3 --eeg_length 30 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --height 64 --width 128 --logdir tiny_vit_21m_512_el30_mixup_50ep --num_workers 6
    
python eval.py --logdir tiny_vit_21m_512_el30_mixup_50ep --batch_size 32 --num_workers 24


# python main.py --seed 10224 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_768_el40_mixup_50ep --num_workers 6

# python main.py --seed 10225 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_768_el40_mixup_50ep --num_workers 6

# python main.py --seed 10226 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_768_el40_mixup_50ep --num_workers 6

# python main.py --seed 10227 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_768_el40_mixup_50ep --num_workers 6

# python main.py --seed 10228 --model_name inception_next_tiny.sail_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir inception_next_tiny_768_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir inception_next_tiny_768_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 5224 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 64 --width 128 --logdir swinv2_tiny_window16_256_512_el40_mixup_50ep --num_workers 6

# python main.py --seed 5225 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 64 --width 128 --logdir swinv2_tiny_window16_256_512_el40_mixup_50ep --num_workers 6

# python main.py --seed 5226 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 64 --width 128 --logdir swinv2_tiny_window16_256_512_el40_mixup_50ep --num_workers 6

# python main.py --seed 5227 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 64 --width 128 --logdir swinv2_tiny_window16_256_512_el40_mixup_50ep --num_workers 6

# python main.py --seed 5228 --model_name swinv2_tiny_window16_256.ms_in1k --drop_path_rate 0.3 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 64 --width 128 --logdir swinv2_tiny_window16_256_512_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir swinv2_tiny_window16_256_512_el40_mixup_50ep --batch_size 32 --num_workers 24


