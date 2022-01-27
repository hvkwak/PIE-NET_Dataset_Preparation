#!/bin/bash

# Run like this: ./run_visualizer.sh visualizer_prep.py 1.mat
filename=$1
dataname=$2
DIR="$dataname"

if [ "$filename" = "visualizer_prep.py" ]; then
	if [ "$(ls -A $DIR)" ]; then
     		echo "file exists: ${filename}. Run this."
	        python3 ${filename} ${dataname}
	else
	        echo "file doesnt exist. download this first:"
	        echo "Type in ID and password"
	        scp hyovin.kwak@dgx01.pro2future.at:/raid/home/hyovin.kwak/PIE-NET_Dataset_Preparation/${filename} ./
	        python3 ${filename} ${dataname}
	fi
fi

if [ "$filename" = "visualizer_results.py" ]; then
	if [ "$(ls -A $DIR)" ]; then
		echo "file exists: ${filename}. Run this."
		python3 ${filename} ${dataname}
	else
                echo "file doesnt exist. download this first:"
                echo "Type in ID and password"
                scp hyovin.kwak@dgx01.pro2future.at:/raid/home/hyovin.kwak/PIE-NET/main/test_results/${filename} ./
                python3 ${filename} ${dataname}
	if
fi
