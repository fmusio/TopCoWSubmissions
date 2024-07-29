"""
Script for running inference on angiographic images and get the multi-class CoW segmentations.

Running script:
    1) Install all prerequisites (see readme.md)
    2) Place your images in the correct input folder, either ./input/head-mr-angio/ or ./input/head-ct-angio/
    3) Specify the track ('mr' or 'ct') under TODO 
    5) Run the script

Output: CoW multi-class segmentation masks saved in the ./output/images/cow-multiclass-segmentation/ folder.
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os
import shutil
import pickle
import json
import subprocess

#######################################################################################
# TODO: First, choose your track!
# track is either 'ct' or 'mr
TRACK = 'ct'

# Optional TODO: Specify name of nnDet detection  and nnUNet segmentation environments
DET_ENV = '.detenv' # default value
SEG_ENV = '.segenv' # default value
# End of TODO
#######################################################################################


class CoWSegBaseline():

    def __init__(self, track: str, top_num_preds: int=10, box_buffer: int=2, det_threshold: float=0.6, folds_seg: str='0 1 2 3 4',
                 det_env_name: str='.detvenv', seg_env_name: str='.segvenv'):
        """
        Args:
            track (str): The track to run the inference on. Either 'mr' or 'ct'.
            top_num_preds (int, optional): The top number of detection predictions to use for final box prediction. Defaults to 10.
            box_buffer (int, optional): Include buffer on all sides for final box cropping. Defaults to 2.
            det_threshold (float, optional): Detection predictions excluded if score smaller than this threshold. Defaults to 0.6.
            folds_seg (str, optional): The folds to use for segmentation. Defaults to '0 1 2 3 4' which is the ensemble of all 5 folds.
            det_env_name (str, optional): The name of the detection environment. Defaults to '.detenv'.
            seg_env_name (str, optional): The name of the segmentation environment. Defaults to '.segenv'.
        """
        self.track = track.lower()
        assert self.track == 'mr'  or self.track == 'ct', "No valid track! Must be 'mr' or 'ct'."

        # Specifiy input/output interface
        # NOTE: Input interface depends on the track
        if self.track == 'mr':
            self.input_path = Path('input/head-mr-angio/')
        elif self.track == 'ct':
            self.input_path = Path('input/head-ct-angio/')
        
        self.output_path = Path('output/images/cow-multiclass-segmentation/')

        # configs for inference with detection model
        self.datadir_detection = Path('nnDet/input')
        self.datadir_detection.mkdir(parents=True, exist_ok=True)
        self.modeldir_detection = Path('nnDet/model')
        self.model_detection = 'RetinaUNetV001_D3V001_3d' 

        if self.track == 'mr':
            self.fold_detection = '1' 
            self.task_detection = 'Task114_CoWDetMRWholeData'
        elif self.track == 'ct':
            self.fold_detection = '4'
            self.task_detection = 'Task115_CoWDetCTWholeData'

        self.top_num_preds = top_num_preds #top number of predictions to use for final box prediction
        self.box_buffer = box_buffer #include buffer on all sides for final box
        self.threshold = det_threshold #predictions excluded if score smaller than this threshold

        # configs for inference with segmentation model
        self.modeldir_segmentation = Path('nnUNet/model')
        if self.track == 'mr':
            self.model_segmentation = 'Dataset112_MRMulSegWholeData'
        elif self.track == 'ct':
            self.model_segmentation = 'Dataset113_CTMulSegWholeData'

        self.input_segmentation = Path('nnUNet/input')
        output_segmentation = Path('nnUNet/output')
        self.output_segmentation_raw = output_segmentation / 'raw'
        self.output_segmentation_raw.mkdir(parents=True, exist_ok=True)
        self.output_segmentation_pp = output_segmentation / 'post-processed'
        self.output_segmentation_pp.mkdir(parents=True, exist_ok=True)
        self.output_segmentation_final = output_segmentation / 'final'
        self.output_segmentation_final.mkdir(parents=True, exist_ok=True)
        self.fold_segmentation = folds_seg 

        # Environment names used in shell scripts
        self.detenv = det_env_name
        self.segenv = seg_env_name

    
    def create_final_det_box(self, input_img_path: Path) -> dict:
        """
        Load predictions, create final box prediction as a composition of the top_num_preds predictions and optionally save roi as json 

        Args:
            input_img_path (Path): The path to the input image file.

        Returns:
            dict: The final prediction box
        """
        
        if self.fold_detection == '-1':
            fold = 'consolidated'
        else:
            fold = 'fold' + self.fold_detection
        
        # load predictions pickle file
        pred_path = self.modeldir_detection / self.task_detection / self.model_detection / fold / 'test_predictions'
        pred_file = pred_path / input_img_path.name.replace('_0000.nii.gz', '_boxes.pkl')
        with open(pred_file, 'rb') as f:
            data = pickle.load(f)

        pred_scores = data['pred_scores']
        pred_boxes = data['pred_boxes']
        idx = np.argsort(pred_scores)

        prediction_meta = {}

        X_min, X_max, Y_min, Y_max, Z_min, Z_max = [], [], [], [], [], []

        # loop over top best predictions
        num_predictions = len(pred_boxes)
        if num_predictions < self.top_num_preds:
            top_preds = num_predictions
            print(f'Not enough predictions! Creating final box from top {num_predictions} (< {self.top_num_preds}) predictions')
        else:
            top_preds = self.top_num_preds
            print(f'Creating final box from top {self.top_num_preds} predictions')
        for k in range(1, top_preds+1):
            box = pred_boxes[idx[-k]]
            score = pred_scores[idx[-k]]
            print("Idx: ", int(idx[-k]), " score: ", score)

            # Only append box if certain threshold score is reached
            if score > self.threshold:
                X_min.append(int(box[4]))
                X_max.append(int(box[5]))
                Y_min.append(int(box[1]))
                Y_max.append(int(box[3]))
                Z_min.append(int(box[0]))
                Z_max.append(int(box[2]))
        
        # Final box is the union of the top_num_preds boxes
        location = (min(X_min)-self.box_buffer, min(Y_min)-self.box_buffer, min(Z_min)-self.box_buffer)
        endpt = (max(X_max)+self.box_buffer, max(Y_max)+self.box_buffer, max(Z_max)+self.box_buffer)
        size = (endpt[0]-location[0]+1, endpt[1]-location[1]+1, endpt[2]-location[2]+1)

        prediction_meta = {
            "top_num_preds": self.top_num_preds,
            "location": location,
            "size": size,
            "endpoint": endpt
        }

        print(f'final prediction:')
        print(prediction_meta)
        print('\n')

        return prediction_meta

    def crop_image_for_segmentation(self, input_img_path: Path, remove_old_rois: bool=False):
        """
        Crops image using final prediction box and saves it in nnUNet folder for segmentation

        Args:
            input_img_path (Path): The path to the input image file.
            remove_old_rois (bool, optional): Whether to remove old ROI json files. Defaults to False.
        """
        
        prediction_meta = self.create_final_det_box(input_img_path)
        end = prediction_meta["endpoint"]
        start = prediction_meta["location"]

        img = sitk.ReadImage(input_img_path)
        img_size = img.GetSize()

        # # Can lead to errors if start or end is outside of image!
        # roi_slice = (
        #     slice(start[0], end[0] + 1),
        #     slice(start[1], end[1] + 1),
        #     slice(start[2], end[2] + 1),
        # )

        # Ensure that start and end are within image boundaries
        start_x, start_y, start_z = max(0, start[0]), max(0, start[1]), max(0, start[2])
        end_x, end_y, end_z = min(img_size[0]-1, end[0]), min(img_size[1]-1, end[1]), min(img_size[2]-1, end[2])
        
        roi_slice = (
            slice(start_x, end_x + 1),
            slice(start_y, end_y + 1),
            slice(start_z, end_z + 1),
        )

        # crop img
        img = img[roi_slice]

        # create folder for cropped images
        output_folder_cropped_img = self.input_segmentation / 'image'
        output_folder_cropped_img.mkdir(parents=True, exist_ok=True)

        # removing old images
        for f in os.listdir(output_folder_cropped_img):
            os.remove(output_folder_cropped_img / f)
        
        # removing old ROI json files
        if remove_old_rois:
            for f in os.listdir(output_folder_roi):
                os.remove(output_folder_roi / f)

        # save img to nnUNet input folder     
        output_path_cropped_img = output_folder_cropped_img / input_img_path.name
        sitk.WriteImage(img, output_path_cropped_img, useCompression=True)

        # save json for final padding
        prediction_meta["file"] = input_img_path.name
        output_folder_roi = self.input_segmentation / 'roi'
        output_folder_roi.mkdir(parents=True, exist_ok=True)
        
        save_path_roi_json = output_folder_roi / input_img_path.name.replace('.nii.gz', '.json')
        with open(save_path_roi_json, 'w') as fp:
            json.dump(prediction_meta, fp)

        return img

    
    def pad_prediction(self, input_img_path: Path):
        """
        Padding prediction to shape of input image and flipping labels if necessary (nnUNet mirroring!)

        Args:
            input_img_path (Path): The path to the input image file.
        """
        path_roi = self.input_segmentation / 'roi'
        roi_file = path_roi / input_img_path.name.replace('.nii.gz', '.json')


        with open(os.path.join(roi_file), 'r') as f:
            dict = json.load(f)
            
        img = sitk.ReadImage(input_img_path)
        shape = img.GetSize()

        mask_array = np.zeros(shape, dtype=np.uint8)

        pred_file = self.output_segmentation_pp / input_img_path.name.replace('_0000.nii.gz', '.nii.gz')
        pred = sitk.ReadImage(pred_file)
        
        # orientation is (z,y,x)
        pred_arr = sitk.GetArrayFromImage(pred)

        pred_arr = np.moveaxis(pred_arr, (0,2), (2,0))

        location = dict['location']
        endpts = dict['endpoint']
        x_min, y_min, z_min = int(location[0]), int(location[1]), int(location[2])
        x_max, y_max, z_max = int(endpts[0]), int(endpts[1]), int(endpts[2]) 

        mask_array[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = pred_arr

        # For nnUNet label 15 was converted to 13. Convert back to 15
        mask_array[mask_array == 13] = 15

        # Check and flip L and R if necessary
        # NOTE: Necessary since mirroring was not turned off for nnUNet!
        LMCA = np.where(mask_array == 7)[0]
        RMCA = np.where(mask_array == 5)[0]
        LPCA = np.where(mask_array == 3)[0]
        RPCA = np.where(mask_array == 2)[0]
        LICA = np.where(mask_array == 6)[0]
        RICA = np.where(mask_array == 4)[0]
        LACA = np.where(mask_array == 12)[0]
        RACA = np.where(mask_array == 11)[0]
        side_diff_mca, side_diff_ica, side_diff_pca, side_diff_aca = 0, 0, 0, 0
        # Require at least 10 voxels of a certain label to make a decision
        if len(LMCA) > 10 and len(RMCA) > 10:
            side_diff_mca = np.sign(np.mean(LMCA) - np.mean(RMCA))
        if len(LICA) > 10 and len(RICA) > 10:
            side_diff_ica = np.sign(np.mean(LICA) - np.mean(RICA))
        if len(LPCA) > 10 and len(RPCA) > 10:
            side_diff_pca = np.sign(np.mean(LPCA) - np.mean(RPCA))
        if len(LACA) > 10 and len(RACA) > 10:
            side_diff_aca = np.sign(np.mean(LACA) - np.mean(RACA))
        
        sum_sides = side_diff_mca + side_diff_ica + side_diff_pca + side_diff_aca

        if sum_sides < 0:
            # flip PCAs
            mask_array[mask_array == 2] = 102
            mask_array[mask_array == 3] = 2
            mask_array[mask_array == 102] = 3
            # flip ICAs
            mask_array[mask_array == 4] = 104
            mask_array[mask_array == 6] = 4
            mask_array[mask_array == 104] = 6
            # flip MCAs
            mask_array[mask_array == 5] = 105
            mask_array[mask_array == 7] = 5
            mask_array[mask_array == 105] = 7
            # flip Pcoms
            mask_array[mask_array == 8] = 108
            mask_array[mask_array == 9] = 8
            mask_array[mask_array == 108] = 9
            # flip ACAs
            mask_array[mask_array == 11] = 111
            mask_array[mask_array == 12] = 11
            mask_array[mask_array == 111] = 12

        #reorient (x,y,z) to (z,y,x) for sitk.GetImageFromArray
        mask_array = np.moveaxis(mask_array, (0,2), (2,0))

        final_mask = sitk.GetImageFromArray(mask_array)
        final_mask.CopyInformation(img)
        
        self.output_segmentation_final.mkdir(parents=True, exist_ok=True)
        final_mask_path = self.output_segmentation_final / input_img_path.name

        sitk.WriteImage(final_mask, final_mask_path, useCompression=True)
    
    def inference_detection(self, input_img_path: Path):
        """
        Running inference using the nnDet detection model

        Args:
            input_img_path (Path): The path to the input image file.
        """
        img = sitk.ReadImage(input_img_path)
        filename = input_img_path.name
        path_task = self.datadir_detection / self.task_detection
        path_raw = path_task / 'raw_splitted/'

        path_img = path_raw / 'imagesTs'
        path_img.mkdir(parents=True, exist_ok=True)

        # removing old images before copying new one 
        for f in os.listdir(path_img):
            os.remove(path_img / f)
        
        # recursively remove preprocessed folder as well
        path_preprocessed = path_task / 'preprocessed'
        if os.path.exists(path_preprocessed):
            shutil.rmtree(path_preprocessed)
        
        # copy image
        path = path_img / filename
        print(f'copying image to {path}...')
        sitk.WriteImage(img, path, useCompression=True)
        print('done!')
    
        # removing old detection predictions
        path_preds = self.modeldir_detection / self.task_detection / self.model_detection
        fold = 'fold' + self.fold_detection
        path_preds = path_preds / fold
        
        test_preds = path_preds / 'test_predictions'
        if os.path.exists(test_preds):
            shutil.rmtree(test_preds) 

        # Calling bash script that runs actual inference command 
        # NOTE: Args must be cast to string
        subprocess.check_call(['./inference_detection.sh', self.task_detection, self.model_detection, self.fold_detection, self.detenv])

        # Cropping image to ROI for segmentation
        self.crop_image_for_segmentation(input_img_path, remove_old_rois=False)
    
    def inference_segmentation(self, input_img_path: Path):
        """
        Running inference using the nnUNet segmentation model

        Args:
            input_img_path (Path): The path to the input image file.
        """
        # Removing old predictions
        for f in os.listdir(self.output_segmentation_raw):
            os.remove(self.output_segmentation_raw / f)
        
        for f in os.listdir(self.output_segmentation_pp):
            os.remove(self.output_segmentation_pp / f)
        
        for f in os.listdir(self.output_segmentation_final):
            os.remove(self.output_segmentation_final / f)
        
        # Calling bash script that runs actual inference command 
        # NOTE: Args must be cast to string
        subprocess.check_call([
            './inference_segmentation.sh', 
            self.model_segmentation, 
            str(self.modeldir_segmentation),
            str(self.input_segmentation), 
            str(self.output_segmentation_raw), 
            str(self.output_segmentation_pp), 
            self.fold_segmentation,
            self.segenv
        ])

        self.pad_prediction(input_img_path)
        
        # copy prediction to final output path
        path_final_pred =self.output_segmentation_final
        final_pred = path_final_pred / input_img_path.name
        
        self.output_path.mkdir(parents=True, exist_ok=True)
    
        shutil.copy(final_pred, os.path.join(self.output_path, final_pred.name))

    
    def run_inference(self):
        """
        Run the inference pipeline for each case in the specified input folder
        """
        # Working with nifti images
        input_images = sorted(self.input_path.glob("*.nii.gz")) # pick all nifti images in the input folder

        # loop through all cases in the input folder
        for image in input_images:
            print(f'\nRunning inference for {image.name}')
            self.inference_detection(image)
            self.inference_segmentation(image)
            print(f'Inference for {image.name} done!')


if __name__ == "__main__":

    # NOTE: TRACK, DET_ENV and SEG_ENV are specified at the top of the script
    cowsegbase = CoWSegBaseline(TRACK, det_env_name=DET_ENV, seg_env_name=SEG_ENV)
    cowsegbase.run_inference()

