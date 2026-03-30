@echo off
call conda activate sonarsplat
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set MSSdk=1
set CUDA_HOME=C:\Users\omviz\.conda\envs\sonarsplat
set TORCH_CUDA_ARCH_LIST=8.6
set BUILD_NO_CUDA=0
set PATH=C:\Users\omviz\.conda\envs\sonarsplat\bin;%PATH%
cd /d D:\Lockheed2026\sonar_splat
pip install --no-build-isolation -e . 2>&1
