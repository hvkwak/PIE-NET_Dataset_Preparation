#!/bin/bash
grep -h "Type" ./log/*generate*.txt | sort | uniq -c | sort -nr
