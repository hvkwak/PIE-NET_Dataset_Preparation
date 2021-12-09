# PIE-NET_Dataset_Preparation
This set of codes generate the datasets suitable for PIE-NET: Parametric Inference of Point Cloud Edges (Wang et al. 2020) from ABC Dataset(Koch et al. 2019). This may eventually create a dataset size of > 10K out of 1M CAD Models based on instructions in [Dataset Preparation](https://github.com/wangxiaogang866/PIE-NET), if you are about to use all ABC Datasets. I personally applied more filtering rules accordingly. Results from Section 3.1 of the paper were reproducible. 

## Usage
Download *.obj and *.yml files from [ABC Dataset](https://deep-geometry.github.io/abc-dataset/)

## Descriptions / Usage
You may find some of these codes useful. Please check the details and arguments accordingly.

make_list.sh - generates the a text file of filenames.
main_multiple_screens.sh - runs main.sh in multiple screens in linux
main.sh - runs main.py
main2.py - there are some datasets that are not size of 64 after running main.py. This brings altogether pack rest datasets to be size of 64. Rests will be discarded.