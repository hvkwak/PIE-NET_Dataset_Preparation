#!/bin/bash

# run this script e.g.: ./run_main.sh 0000 128
echo "ENTER the prefix number: 0000-0099"
echo "argument ${1}"
#read folder_num
folder_num=${1}
#echo "$folder_num"


list_obj="/raid/home/hyovin.kwak/all/obj/${folder_num}_list_obj.txt"
list_yml="/raid/home/hyovin.kwak/all/obj/${folder_num}_list_features.txt"
log_dir="/raid/home/hyovin.kwak/PIE-NET_Dataset_Preparation/log/"

python3 main.py $list_obj $list_yml $folder_num $log_dir ${2} ${3}
