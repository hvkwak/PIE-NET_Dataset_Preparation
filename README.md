# PIE-NET_Dataset_Preparation
This set of codes generate the datasets suitable for PIE-NET: Parametric Inference of Point Cloud Edges (Wang et al. 2020) from ABC Dataset(Koch et al. 2019). This will create a dataset size of > 10K out of 1M CAD Models based on instructions in [Dataset Preparation](https://github.com/wangxiaogang866/PIE-NET), if you are about to use all chunks of ABC Dataset. I personally applied more filtering rules accordingly. 

## Please note
Results from Section 3.1 of the paper were reproducible. No Warranty, No Liability.

## How to use / Useful Descriptions
You may find some of these codes useful. First download *.obj and *.yml files from [ABC Dataset](https://deep-geometry.github.io/abc-dataset/), then run 

```bash
make_list.sh
```
which generates text files of lists to prepare for further steps. Then run other codes accordingly. Please check the details, directories and arguments in advance. <br />

make_list.sh generates the a text file of filenames. <br />
main_multiple_screens.sh runs main.sh in multiple screens in linux <br />
main.sh runs main.py <br />
main2.py there are some datasets that are not size of 64 after running main.py. This brings altogether pack rest datasets to be size of 64. Rests will be discarded.<br />
