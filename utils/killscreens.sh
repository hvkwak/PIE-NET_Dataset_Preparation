#!/bin/bash
for scr in $(screen -ls | awk '{print $1}'); do screen -S $scr -X kill; done
