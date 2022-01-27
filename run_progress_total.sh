#!/bin/bash
echo "Successfully generated:"
grep "Ok" ./log/*generate* | wc -l

echo "Processed objects:"
grep "Processing" ./log/*generate* | wc -l
