#!/bin/bash
export JULIA_PROJECT=$(realpath $(dirname $0)/..)
# python $JULIA_PROJECT/src/rnn.py --epochs=1 2> /dev/null
# python $JULIA_PROJECT/examples/ort1d.py 2> /dev/null
# python $JULIA_PROJECT/examples/ort2d.py 2> /dev/null
# python $JULIA_PROJECT/examples/inception.py 2> /dev/null
# julia $JULIA_PROJECT/examples/ueaucr.jl
# find job_ueaucr -name 'julia.sh' | xargs -n 1 -P ${1-1} bash
julia $JULIA_PROJECT/examples/longmem.jl
find job_longmem -name 'julia.sh' | xargs -n 1 -P ${1-1} bash