#!/bin/bash

# make_list.sh - copies *.yml files to obj folders, generates the list of files of .obj and .yml files.
#
# Instructions:
# 1. save all the .obj files in .../all/obj
# 2. set the regex numbers in line 10 accordingly
# 3. run this file.
#
#
cd '/raid/home/hyovin.kwak/all/obj/'
ls -d 00[0][0] > 0000_list.txt
list_dir="./0000_list.txt"
while IFS= read -r line1
do
  echo "${line1}" # e.g. "0000"
  cp /raid/home/hyovin.kwak/all/feat/${line1}/*.yml ${line1}
  cd ${line1}
  ls *.yml | awk -F '_features' '{print $1}' > ../${line1}_list_yml.txt
  cd ..
  filename="${line1}_list_yml.txt"
  echo $filename
  # current dir: ...all/obj
  # per folder e.g. 0000, check all the files:
  while IFS= read -r line2
  do
    if ls ${line1}/${line2}*.obj &>/dev/null; then
	    echo "Found"
    else
	    echo "Not found."
	    rm ${line1}/${line2}*.yml
    fi
  done < "$filename"
  ls /raid/home/hyovin.kwak/all/obj/${line1}/*.obj > ${line1}_list_obj.txt
  ls /raid/home/hyovin.kwak/all/obj/${line1}/*.yml > ${line1}_list_features.txt
done < "$list_dir"
