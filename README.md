## Overview
This set of codes generates the datasets suitable for PIE-NET: Parametric Inference of Point Cloud Edges (Wang et al. 2020) from ABC Dataset(Koch et al. 2019), which may be size of ca. 2K out of 750K CAD models based on instructions in [Dataset Preparation](https://github.com/wangxiaogang866/PIE-NET). Note that there have been many unsuitable models regarding the difficulty level that are supposed to be excluded from the dataset. It is recommended to check the generated dataset once again after running the `main.py`, which will unfortunately still have some unsuitable models.

## How to generate
You may find some of codes in `utils` useful. First download `*.obj` and `*.yml` files from [ABC Dataset](https://deep-geometry.github.io/abc-dataset/), then run 
```bash
make_list.sh
```
which generates text files of lists to prepare for further steps. Then run other codes accordingly. To start Dataset generation, run
```bash
main.sh
```
where you will enter an additional argument to select a "chunk number" of the Dataset. You may also find `main_multiple_screens.sh` useful if you are willing to do this in mutliple screens and processors. Please check the details, directories and arguments in advance. <br />

`make_list.sh` generates the a text file of filenames. <br />
`main_multiple_screens.sh` runs `main.sh` in multiple screens in linux <br />
`main.sh` runs `main.py` <br />

## Please note
Results from Section 3.1 and 3.2. of the paper were reproducible. Note that this code will generate objects per chunk. You may have to pack all the pieces together so that the dataset is ready for training neural networks. Please understand that some codes are not efficient enough, or copy-and-paste.
