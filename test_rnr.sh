# novel view synthesis
python test_rnr.py --lighting_type train --calib_dir _/test_seq/spiral_step720 \
--checkpoint_dir [directory of checkpoint] \
--checkpoint_name [file name of checkpoint] \
--img_size 512 --sampling_pattern all --gpu_id 0

# free-viewpoint relighting
python test_rnr.py --lighting_type SH --lighting_idx 1 --calib_dir _/test_seq/spiral_step720 \
--checkpoint_dir [directory of checkpoint] \
--checkpoint_name [file name of checkpoint] \
--img_size 512 --sampling_pattern all --gpu_id 0