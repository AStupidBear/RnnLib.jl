#!/bin/bash
export JULIA_PROJECT=$(realpath $(dirname $0)/..)
# python $JULIA_PROJECT/src/rnn.py --epochs=1 2> /dev/null
# python $JULIA_PROJECT/examples/ort1d.py 2> /dev/null
# python $JULIA_PROJECT/examples/ort2d.py 2> /dev/null
# python $JULIA_PROJECT/examples/inception.py 2> /dev/null
julia ucr.jl && ls $dir/*/julia.sh | xargs -n 1 -P ${1-5} bash