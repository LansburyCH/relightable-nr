python train_dnr.py --data_root ./data/material_sphere --img_dir _/rgb0 --img_size 512 \
--obj_fp _/mesh.obj \
--batch_size 1 --gpu_id 0 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 1000 \
--exp_name example