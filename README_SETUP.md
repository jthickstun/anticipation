### Connect to EC2 Container
In a terminal run: `ssh -i "mg-bso.pem" ubuntu@ec2-54-225-45-227.compute-1.amazonaws.com`


Setup Conda
```shell
conda init bash
conda update -n base -c conda-forge conda
```

Active PyTorch
```shell
source activate pytorch
conda config --set auto_activate_base true
```

Setup Riffusion and Conda Env for Riffusion
```shell
git clone https://github.com/riffusion/riffusion.git
cd riffusion

conda create --name riffusion python=3.x
conda activate riffusion

python -m pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```


### Test for cuda
Open a Python shell (`python`) and run:
```
import torch
torch.cuda.is_available()
```

### Run Riffusion UI Playground
```shell
python -m riffusion.streamlit.playground
```

Add outbound rules for HTTP and HTTPS 0.0.0.0/0
Add inbound rules for 8501 0.0.0.0/0 or our IPs



Other commands
```shell
sudo apt-get update
sudo apt-get upgrade
```


### Install Jupyter Notebook on EC2 Container
```shell
sudo apt install jupyter
conda activate anticipation
jupyter lab --ip 0.0.0.0
```

```shell
conda active anticipation
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```


!git clone https://github.com/jthickstun/anticipation.git
!pip install ./anticipation
!pip install -r anticipation/requirements.txt


After setup, to launch and run the Jupyter Lab for Anticipation
```shell
cd anticipation/
conda activate anticipation
jupyter lab --ip 0.0.0.0
```

in browser to access Jupyter Lab
url: `'http://' + ip_address + ':8888/lab/tree/stage1.ipynb'`
password: `bso-mg-1!`
