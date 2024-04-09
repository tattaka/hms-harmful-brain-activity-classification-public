# python main.py --seed 2424 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 0 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2425 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 1 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2426 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 2 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2427 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 3 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 2428 --model_name resnetrs50 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --batch_size 8 --fold 4 --gpus 4 --epochs 50 --mixup_p 0.5 --mixup_alpha 0.2 \
#     --height 96 --width 192 --logdir resnetrs50_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir resnetrs50_384_el40_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak.py --logdir resnetrs50_384_el40_mixup_50ep --batch_size 64 --num_workers 24
# python make_pseudo_label_wo_leak_train.py --logdir resnetrs50_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 3424 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --batch_size 16 --fold 0 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3425 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --batch_size 16 --fold 1 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3426 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40  --fmin 0.1 --fmax 40\
#     --lr 1e-3 --batch_size 16 --fold 2 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3427 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --batch_size 16 --fold 3 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 3428 --model_name resnetrs101 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --batch_size 16 --fold 4 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir resnetrs101_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir resnetrs101_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 4424 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4425 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4426 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4427 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 4428 --model_name caformer_b36.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir caformer_b36_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir caformer_b36_384_el40_mixup_50ep --batch_size 64 --num_workers 24


python main.py --seed 5424 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5425 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5426 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5427 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6

python main.py --seed 5428 --model_name convnext_large.fb_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
    --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
    --height 96 --width 192 --logdir convnext_large_384_el40_mixup_50ep --num_workers 6
    
python eval.py --logdir convnext_large_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 6424 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6425 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6426 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6427 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6

# python main.py --seed 6428 --model_name davit_base.msft_in1k --drop_path_rate 0.2 --eeg_length 40 --fmin 0.1 --fmax 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --win_length 256 \
#     --height 96 --width 192 --logdir davit_base_384_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir davit_base_384_el40_mixup_50ep --batch_size 64 --num_workers 24


# python main.py --seed 7424 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 0 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7425 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 1 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7426 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 2 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7427 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 3 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6

# python main.py --seed 7428 --model_name inception_next_base.sail_in1k_384 --drop_path_rate 0.2 --eeg_length 40 \
#     --lr 1e-3 --backbone_lr 5e-4 --batch_size 8 --fold 4 --gpus 4 --epochs 25 --mixup_p 0.5 --mixup_alpha 0.2 --fmin 0.1 --fmax 40 --win_length 256 \
#     --height 96 --width 192 --logdir inception_next_base_el40_mixup_50ep --num_workers 6
    
# python eval.py --logdir inception_next_base_el40_mixup_50ep --batch_size 64 --num_workers 24