<div align="center">
  <h1 align="center">:alembic: S-Prompts Learning with Pre-trained Transformers: Extended Implementation</h1>
  <p align="center"> 
  </p>
</div>

<div align="center">
    <a href=#original-project>Original project</a>
    •
    <a href=#modifications-and-enhancements>Changes</a>
    •
    <a href=#environment-setup>Env setup</a>
    •
    <a href=#dataset-preparation>Dataset</a>
    •
    <a href=#training >Training</a>
    •
    <a href=#evaluation >Evaluation</a>
</div>
<br>

This repository is a personal extension of the original project "S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning", originally presented at NeurIPS 2022. My version includes some minor fixes and enhancements to the original implementation.

## Original Project

The original project, developed by Yabin Wang, Zhiwu Huang, and Xiaopeng Hong, introduced a novel paradigm in Domain Incremental Learning (DIL) using pre-trained transformers. The official implementation and paper can be found here: [[Original Repository](https://github.com/iamwangyabin/S-Prompts) | [Paper](https://openreview.net/pdf?id=ZVe_WeMold)].

## Modifications and Enhancements

>Note: I have only performed and tested CDDB experiments using [cddb_slip.json](configs/cddb_slip.json). **Pull requests are welcome!**

In this fork, I have made the following changes:

- Include the requirements.txt for pip - not available in the original repo 
- Add eval directly in this repo without relying on the original [evaulation repo](https://github.com/iamwangyabin/SPrompts_eval) - eval issue multiplication with 0 at this [line](https://github.com/iamwangyabin/SPrompts_eval/blob/9e63db433650102b51d1232d7aff4a56dbeb3d59/eval.py#L131) 
- Save model as `state_dict()` instead of the full model
- Bug fixes related to one GPU usage and ML/DL frameworks args
- Introduction of Early stopping option
- Move config parameters in the JSON file
- Tested on PyTorch 2.1.0 and CUDA 12.1
- Added scenario (CDDB Hard or OOD) and compression option
- Added label smoothing option

> Note: As the original project was split in two different repository, one for training and one for evaluation, I have decided not to pull request on the original repository but made a separate one.


## Enviroment setup
Create the virtual environment for S-Prompts. Tested on Python 3.9 and NVIDIA GPU A30 with MIG partition 6 GB memory.
```
python -m pip install -r requirements.txt
```

## Dataset preparation
Please refer to the following links to download three standard domain incremental learning benchmark datasets. 

[CDDB](https://github.com/Coral79/CDDB)  
[CORe50](https://vlomonaco.github.io/core50/index.html#dataset)  
[DomainNet](http://ai.bu.edu/M3SDA/)  

Unzip the downloaded files, and you will get the following folders.
```
CDDB
├── biggan
│   ├── train
│   └── val
├── gaugan
│   ├── train
│   └── val
├── san
│   ├── train
│   └── val
├── whichfaceisreal
│   ├── train
│   └── val
├── wild
│   ├── train
│   └── val
... ...
```

```
core50
└── core50_128x128
    ├── labels.pkl
    ├── LUP.pkl
    ├── paths.pkl
    ├── s1
    ├── s2
    ├── s3
    ...
```

```
domainnet
├── clipart
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── clipart_test.txt
├── clipart_train.txt
├── infograph
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
├── infograph_test.txt
├── infograph_train.txt
├── painting
│   ├── aircraft_carrier
│   ├── airplane
│   ... ...
... ...
```


## Training

Please change the `data_path` in the config files to the locations of the datasets.

### CDDB
```
python main.py --config configs/cddb_slip.json
```

### CORe50

_Not tested, contribute!_

### DomainNet

_Not tested, contribute!_

## Evaluation
```
python eval.py --config configs/cddb_slip.json
```

## License

Please check the MIT  [license](./LICENSE) that is listed in this repository.

## Acknowledgments

My work builds upon the foundation laid by Yabin Wang, Zhiwu Huang, Xiaopeng Hong, and the contributors to the original project. I extend my gratitude to them for their groundbreaking work.