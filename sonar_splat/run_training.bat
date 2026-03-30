@echo off
call conda activate sonarsplat
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
set CUDA_HOME=C:\Users\omviz\.conda\envs\sonarsplat
set TORCH_CUDA_ARCH_LIST=8.6
set PATH=C:\Users\omviz\.conda\envs\sonarsplat\bin;%PATH%
cd /d D:\Lockheed2026\sonar_splat
python examples/sonar_simple_trainer.py prune_only --batch_size 1 --camera_model ortho --data_dir data/sonarsplat_dataset/concrete_piling_3D --result_dir results/test_run --data_factor 1 --disable_viewer --init_type predefined --init_num_pts 100000 --init_opa 0.9 --init_scale 0.01 --max_steps 100 --near_plane -10 --far_plane 10 --test_every 8 --train --render_eval --sh_degree 3 --tb_every 50 --tb_save_image --skip_frames 1 --start_from_frame 0 --end_at_frame 10000 2>&1
