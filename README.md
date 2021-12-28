# Overview
This set of codes generate the datasets suitable for PIE-NET: Parametric Inference of Point Cloud Edges (Wang et al. 2020) from ABC Dataset(Koch et al. 2019). This may create a dataset size of > 2.6K out of 1M CAD Models based on instructions in [Dataset Preparation](https://github.com/wangxiaogang866/PIE-NET), if you are about to use all chunks of ABC Dataset. More "filtering" rules were added accordingly to fulfill the requirements as many as possible.

## Please note
Results from Section 3.1 of the paper were reproducible. No Warranty, No Liability.

## How to use / Brief Descriptions
You may find some of codes in `utils` useful. First download `*.obj` and `*.yml` files from [ABC Dataset](https://deep-geometry.github.io/abc-dataset/), then run 
```bash
make_list.sh
```
which generates text files of lists to prepare for further steps. Then run other codes accordingly. To start Dataset generation, run
```bash
main.sh
```
where you will enter an additional argument to select a "chunk number" of the Dataset. You may also find `main_multiple_screens.sh` useful if you are willing to do this in mutliple screens. Please check the details, directories and arguments in advance. <br />

`make_list.sh` generates the a text file of filenames. <br />
`main_multiple_screens.sh` runs `main.sh` in multiple screens in linux <br />
main.sh runs `main.py` <br />
main2.py there are some datasets that are not size of 64 after running `main.py`. This pack the "leftover" datasets altogether to be size of 64. The last that failed to form a file size of 64 will be discarded.<br />
