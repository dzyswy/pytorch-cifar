# pytorch-cifar
pytorch cifar


# conda 环境

conda create -n cifar python=3.8
conda env list
conda activate cifar
# win10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia     
conda install tensorboard numpy scipy scikit-learn pandas matplotlib opencv onnx

# ubuntu
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64

conda deactivate
conda env list
conda remove --name cifar --all

