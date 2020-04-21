#!/bin/bash
python ort.py 2> /dev/null
python ort2.py 2> /dev/null
python inception 2 > /dev/null
/bin/rm model.*