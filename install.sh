conda create -n RoGSplat python=3.8
conda activate RoGSplat
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c iopath iopath
conda install pytorch3d -c pytorch3d
pip install -r requirement.txt
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization/
pip install git+https://github.com/AllenXiangX/SnowflakeNet/models/pointnet2_ops_lib