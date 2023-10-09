
# Attentional Graph Neural Network for Parking Slot Detection

![image](https://github.com/Jiaolong/gcn-parking-slot/blob/main/images/animated.gif)

Repository for the paper ["Attentional Graph Neural Network for Parking Slot Detection"](https://arxiv.org/abs/2104.02576).
```
@article{gcn-parking-slot:2020,
  title={Attentional Graph Neural Network for Parking Slot Detection},
  author={M. Chen, J. Xu, L. Xiao, D. Zhao etal},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2021},
  volume={6},
  number={2},
  pages={3445-3450},
  doi={10.1109/LRA.2021.3064270}
}
```

## Requirements

- python 3.6

- pytorch 1.4+

- other requirements: `pip install -r requirements.txt`

## Pretrained models

Two pre-trained models can be downloaded with following links.

| Link      | Code | Description |
| ----------- | ---- | ----------- |
| [Model0](https://pan.baidu.com/s/137ZHZnsEfyaO4yaa5YoBIQ) | bc0a | Trained with ps2.0 subset as in [1]|
| [Model1](https://pan.baidu.com/s/1qogTCwtjGEtR0y-PB4Ibmg)   | pgig  | Trained with full ps2.0 dataset      |

## Prepare data

The original ps2.0 data and label can be found [here](https://github.com/Teoge/DMPR-PS). Extract and organize as follows:

```
├── datasets
│   └── parking_slot
│       ├── annotations (copy testing and training here)
│       ├── ps_json_label (download from DMPR-PS)
│       ├── testing (from ps2.0)
│       └── training (from ps2.0)
```
Then you can use the following command lines to prepare the data, please note that these lines are run in DMPR-PS directory, following its data preparation procedure.
```bash

```
## Train & Test

Export current directory to `PYTHONPATH`:

```bash
export PYTHONPATH=`pwd`
```
or add the following lines to `demo.py`, `train.py`, `test.py` in `tool` directory at the top of file:
```python
import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
```

- demo

```
python3 tools/demo.py -c config/ps_gat.yaml -m cache/ps_gat/100/models/checkpoint_epoch_200.pth
```

- show demo in ps2.0 dataset:
```
python tools/demo.py -c config/ps2_gat.yaml -m checkpoint_epoch_200.pth
```

- show demo in b2 dataset:
```
python tools/demo.py -c config/b2_gat.yaml -m checkpoint_epoch_200.pth
```

- train

```
python3 tools/train.py -c config/ps_gat.yaml
```

- train in ps2.0 dataset

```
python tools/train.py -c config/ps2_gat.yaml
```

- train in b2 dataset

```
python tools/train.py -c config/b2_gat.yaml
```

- test

```
python3 tools/test.py -c config/ps_gat.yaml -m cache/ps_gat/100/models/checkpoint_epoch_200.pth
```

- test in ps2.0 dataset

```
python tools/test.py -c config/ps2_gat.yaml -m checkpoint_epoch_200.pth
```

- test in b2 dataset

```
python tools/test.py -c config/b2_gat.yaml -m checkpoint_epoch_200.pth
```

## References

[1] J. Huang, L. Zhang, Y. Shen, H. Zhang, and Y. Yang, “DMPR-PS: A novel approach for parking-slot detection using directional marking-point regression,” in IEEE International Conference on Multimedia and Expo (ICME), 2019. [code](https://github.com/Teoge/DMPR-PS)
