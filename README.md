# Action classification models
| Mediapipe + ProtoGCN (no norm) | Movenet + ProtoGCN (no norm)|
| --- | --- |
| ![Alt Text](data/coatman_mediapipe.gif) | ![Alt Text](data/coatman_movenet.gif) |

# Python
```bash
pyenv install --list
pyenv install 3.12.4
pyenv versions
pyenv local 3.12.4
```

# Venv + Requirements
```bash
python -m venv .action_env
source .action_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# Launch classification
## Movenet
```shell
export PYTHONPATH=.:protogcn && python classification/run_movenet.py 
```

## Mediapipe
```shell
export PYTHONPATH=.:protogcn && python classification/run_mediapipe.py 
```


# Pretrained Models
All the checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1BLtlGlv19nY6QcYsVyOBo7nBr3iw5cFl?usp=sharing).

# Acknowledgements
This repo is mainly based on [ProtoGCN](https://github.com/firework8/ProtoGCN)