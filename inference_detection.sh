#!/bin/bash

# exit on error
set -e

# check number of params
if [ "$#" -ne 4 ]; then
    echo "inference_detection.sh takes 4 arguments: task, model, fold, env_name"
    echo "Illegal number of parameters"
    exit 1
fi

echo "running inference on detection model with task: $1, model: $2, fold: $3"

#activate the nnDet env
echo ""
echo "activating detection environment $4"
source "./nnDet/$4/bin/activate"
echo "which python3:"
which python3
echo ""

echo ""
echo "Image will be loaded from $det_data"
echo "Model will be loaded from $det_models"

nndet_predict $1 $2 --fold $3
