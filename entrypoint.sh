#!/bin/bash


echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

ls -l $INPUT_DIR
ls -l $OUTPUT_DIR
# python main.py

# when done, watch fs to see if changes happen.
python stat.py