#1 /bin/bash

BASEDIR=$(dirname "$0")
export PYTHONPATH=$BASEDIR:$PYTHONPATH
python $BASEDIR/gpu.py
