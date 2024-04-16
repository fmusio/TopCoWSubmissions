#!/bin/bash

# exit on error
set -e

# check number of params
if [ "$#" -ne 7 ]; then
    echo "inference_segmentation.sh takes 7 arguments: model, model_dir, input, output_raw, output_pp, folds, env_name"
    echo "Illegal number of parameters"
    exit 1
fi

echo "running inference on segmentation model with model: $1 and folds: $6"

# activate the nnUNet env
echo ""
echo "activating segmentation environment $7"
source "./nnUNet/$7/bin/activate"
echo "which python3:"
which python3
echo ""

# Specify folds that you 
nnUNetv2_predict -d $1 -i $3/image -o $4 -f $6 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
nnUNetv2_apply_postprocessing -i $4 -o $5 -pp_pkl_file ${2}/Dataset112_MRMulSegWholeData/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json ${2}/Dataset112_MRMulSegWholeData/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json