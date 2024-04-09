# python main.py --seed 2024 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs50_384_el40_mixup_50ep \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2025 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs50_384_el40_mixup_50ep \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2026 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs50_384_el40_mixup_50ep \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2027 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs50_384_el40_mixup_50ep \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2028 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs50_384_el40_mixup_50ep \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir resnetrs50_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 3524 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs101_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3525 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs101_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3526 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs101_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3527 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs101_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3528 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --batch_size 16 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/resnetrs101_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir resnetrs101_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 4524 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/caformer_b36_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4525 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/caformer_b36_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4526 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/caformer_b36_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4527 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/caformer_b36_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4528 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/caformer_b36_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir caformer_b36_384_el40_mixup_50ep --batch_size 64 --num_workers 24


python main.py --seed 5524 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 \
    --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --resume_dir exp107/convnext_large_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5525 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 \
    --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --resume_dir exp107/convnext_large_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5526 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 \
    --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --resume_dir exp107/convnext_large_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5527 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 \
    --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --resume_dir exp107/convnext_large_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5528 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 \
    --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
    --resume_dir exp107/convnext_large_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6
    
python eval.py --logdir convnext_large_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 6524 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/davit_base_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6525 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/davit_base_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6526 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/davit_base_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6527 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/davit_base_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6528 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/davit_base_384_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir davit_base_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 7524 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/inception_next_base_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7525 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/inception_next_base_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7526 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/inception_next_base_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7527 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/inception_next_base_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7528 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 5e-4 --backbone_lr 2e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --resume_dir exp107/inception_next_base_el40_mixup_50ep --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir inception_next_base_el40_mixup_50ep --batch_size 64 --num_workers 24