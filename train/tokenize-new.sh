#!/bin/bash

data_directory="$1"
data_directory="${data_directory%/}"  # Remove trailing slash if it exists

out_directory="$2" task="$3" 
context="$4"
split="$5"
workers=16

dataset="${data_directory##*/}"
datafile=$out_directory/"$dataset"."$task"."$split".txt

tmpdir="tmp_$(date +%Y%m%d_%H%M%S)"
mkdir -p $out_directory/$tmpdir
python -m train.tokenize-new $data_directory $out_directory/$tmpdir $task $context --workers $workers

echo "Writing to datafile: $datafile"

cat $out_directory/$tmpdir/* > $datafile

if [ "$split" == "train" ]; then
    echo "Shuffling the datafile."
    shuf $datafile -o $datafile
elif [ "$split" == "valid" ]; then
    echo "Writing a small shuffled subsample to $out_directory/staging"
    mkdir -p $out_directory/staging
    smallfile=$out_directory/staging/"$dataset"."$task"."$split"-small.txt
    shuf $datafile -o $smallfile
    head -n 10000 $smallfile > $smallfile.tmp
    mv $smallfile.tmp $smallfile
else
    echo "Datafile marked test. Not shuffling."
fi


# Clean up temporary shard files
rm -r $out_directory/$tmpdir
