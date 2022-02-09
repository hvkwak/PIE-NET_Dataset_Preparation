#!/bin/bash
echo "ENTER the prefix number: 0000-0099"
echo "argument ${1}"
#read folder_num
folder_num=${1}
#echo "$folder_num"


list_obj="/home/pro2future/Documents/all/obj/${folder_num}_list_obj.txt"
list_yml="/home/pro2future/Documents/all/obj/${folder_num}_list_features.txt"
log_dir="/home/pro2future/Documents/PIE-NET_Dataset_Preparation/log/"

python3 main.py $list_obj $list_yml $folder_num $log_dir
