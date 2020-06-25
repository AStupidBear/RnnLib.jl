#!/bin/bash
dir=$(dirname $0)
python $dir/../src/rnn.py --data_path=$HOME/job/train.rnn
python $dir/ort.py 2> /dev/null
python $dir/ort2.py 2> /dev/null
python $dir/inception.py 2> /dev/null
/bin/rm model.*