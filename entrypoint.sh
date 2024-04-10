#!/bin/bash


echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

ls -l $INPUT_DIR
ls -l $OUTPUT_DIR
# Then you can use these variables in your script
# python app.py --input $INPUT_DIR --output $OUTPUT_DIR
python main.py
# when done, watch fs to see if changes happen.
python stat.py