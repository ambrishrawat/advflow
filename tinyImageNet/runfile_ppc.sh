export LD_LIBRARY_PATH=/dccstor/dlw/local-ppc64le/lib/:/usr/local/cuda-7.5/lib64/:/opt/share/lsf-9.1.3/9.1/linux3.13-glibc2.19-ppc64le/lib/
export CUDA_HOME=/usr/local/cuda-7.5/
source /dccstor/dlw/virtual_environments/python3.5-ppc64le/bin/activate
PYTHONPATH=$PYTHONPATH:/dccstor/dlw/code/LearningDeeply python $1
