#!/bin/bash
list_dir="./log/log_progress.txt"
while IFS= read -r line1
do
    #echo ${line1}
    survived_num=$(grep "Ok" ${line1} | wc -l)
    processed_num=$(grep "Processing" ${line1} | awk '/../ {a=$2} END{print a}')
    echo "$survived_num models were successful, currently.. $processed_num"

done < "$list_dir"
