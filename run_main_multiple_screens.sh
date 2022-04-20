#!/bin/bash
echo "Enter start prefix: 0-99"
read start
echo "start prefix: $start"
echo "Enter end prefix > start prefix, end should be 100 to completely finish this."
read end
echo "end prefix: $end"
echo "Enter subsampling points number: 64 or 128"
read sn
echo "subsampling number: $sn"
while [ $end -gt $start ];
do
	#if [ ${start} == 0 ]
	#then
	#   echo "0000"
	#   ./main.sh "0000"
	if [ ${#start} == 1 ]
	then
	   echo "1 digit"
           screen -dmS $start "./run_main.sh" "000$start" "4" "$sn"
        elif [ ${#start} == 2 ]
	then
	   echo "2 digits"
	   screen -dmS $start "./run_main.sh" "00$start" "4" "$sn"
	else
	   echo "digit numbers are wrong. try again."
        fi
	#screen -dmS $start "./main.sh" "$start"
	start=$(($start + 1))
	#echo ${#start}
	#echo 000$start
done
#screen -dmS $prefix "./main.sh" "$prefix"

