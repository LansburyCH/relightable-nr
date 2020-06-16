python train_rnr.py --data_root ./data/material_sphere/ --img_size 512 \
--obj_high_fp _/mesh.obj --obj_low_fp _/mesh_7500v.obj --obj_gcn_fp _/mesh_7500v.obj \
--lp_dir _/light_probe --lighting_idx 0 --lighting_relight_idx 1 \
--batch_size 1 --gpu_id 0,1 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 100 \
--exp_name example
