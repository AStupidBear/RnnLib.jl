#!/bin/bash
dir=$(dirname $0)
python $dir/../src/rnn.py --epochs=1 2> /dev/null
python $dir/ort1d.py 2> /dev/null
python $dir/ort2d.py 2> /dev/null
python $dir/inception.py 2> /dev/null
/bin/rm model.*