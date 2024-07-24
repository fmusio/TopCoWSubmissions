# TopCoW Submissions
This repo contains the inference code for the [TopCoW'23](https://topcow23.grand-challenge.org/) multi-class CoW segmentation submissions of the organizers Fabio Musio and Kaiyuan Yang. The submissions were **ranked 2nd (MRA) and 3rd (CTA)** on our [public leaderboards](https://topcow23.grand-challenge.org/evaluation/finaltest-mra-multiclass/leaderboard/).

Details and results on our TopCoW challenge can be found in the TopCoW summary paper on arXiv: [https://arxiv.org/abs/2312.17670](https://arxiv.org/abs/2312.17670). 
```
@misc{topcowchallenge,
    title={Benchmarking the CoW with the TopCoW Challenge: Topology-Aware Anatomical Segmentation of the Circle of Willis for CTA and MRA},
    author={Kaiyuan Yang and Fabio Musio and Yihui Ma and Norman Juchler and Johannes C. Paetzold and Rami Al-Maskari and Luciano Höher and Hongwei Bran Li and Ibrahim Ethem Hamamci and Anjany Sekuboyina and Suprosanna Shit and Houjing Huang and Chinmay Prabhakar and Ezequiel de la Rosa and Diana Waldmannstetter and Florian Kofler and Fernando Navarro and Martin Menten and Ivan Ezhov and Daniel Rueckert and Iris Vos and Ynte Ruigrok and Birgitta Velthuis and Hugo Kuijf and Julien Hämmerli and Catherine Wurster and Philippe Bijlenga and Laura Westphal and Jeroen Bisschop and Elisa Colombo and Hakim Baazaoui and Andrew Makmur and James Hallinan and Bene Wiestler and Jan S. Kirschke and Roland Wiest and Emmanuel Montagnon and Laurent Letourneau-Guillon and Adrian Galdran and Francesco Galati and Daniele Falcetta and Maria A. Zuluaga and Chaolong Lin and Haoran Zhao and Zehan Zhang and Sinyoung Ra and Jongyun Hwang and Hyunjin Park and Junqiang Chen and Marek Wodzinski and Henning Müller and Pengcheng Shi and Wei Liu and Ting Ma and Cansu Yalçin and Rachika E. Hamadache and Joaquim Salvi and Xavier Llado and Uma Maria Lal-Trehan Estrada and Valeriia Abramova and Luca Giancardo and Arnau Oliver and Jialu Liu and Haibin Huang and Yue Cui and Zehang Lin and Yusheng Liu and Shunzhi Zhu and Tatsat R. Patel and Vincent M. Tutino and Maysam Orouskhani and Huayu Wang and Mahmud Mossa-Basha and Chengcheng Zhu and Maximilian R. Rokuss and Yannick Kirchhoff and Nico Disch and Julius Holzschuh and Fabian Isensee and Klaus Maier-Hein and Yuki Sato and Sven Hirsch and Susanne Wegener and Bjoern Menze},
    year={2024},
    eprint={2312.17670},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2312.17670},
}
```

## Introduction
The Circle of Willis (CoW) is an important anastomotic network of arteries connecting the anterior and posterior circulations of the brain, as well as the left and right cerebral hemispheres.  Due to its centrality, the CoW is commonly involved in pathologies like aneurysms and stroke. Hence, an accurate characterization of the CoW is of great clinical relevance. 

The TopCoW challenge was the first public challenge on CoW angio-architecture extraction and brain vessel segment annotation on two common angiographic imaging modalities, namely magnetic resonance angiography (MRA) and computed tomography angiography (CTA). 
The aim of the challenge was to extract the CoW vessels from 3D angiographic imaging by segmentation of the vessel components. For that purpose, we released a new [dataset](https://topcow23.grand-challenge.org/data/) of joint-modalities, CTA and MRA of the same patient cohort, both with annotations of the anatomy of the CoW.

As organizers, we submitted baseline algorithms for the CoW multi-class segmentation for both MRA and CTA. The inference code and model weights are published in this repo and can be used freely for CoW vessel segmentation tasks.

<p align="center">
  <img src="https://github.com/fmusio/TopCoWSubmissions/blob/main/topcow_segmentation_mr_095.png" width="600" />
</p>


## Method Description
A simple two-stage approach was employed: The [nnDetection](https://github.com/MIC-DKFZ/nnDetection) framework was used to detect and extract custom CoW ROIs based on the binary labels and a 3D [nnUNet](https://github.com/MIC-DKFZ/nnUNet) was employed for the subsequent multi-class segmentation on the ROIs. Additionally, inter-modal registration was used as a data augmentation strategy, registering all the image pairs and thereby doubling the size of the training set for both modalities.

## Usage
### Prerequisites
Clone this repo first.

Both the **nnDetection** and the **nnUNet** frameworks need to be installed. Due to a lack of compatibility of their respective dependencies, two separate environments must be created and activated consecutively. 

Other than that, the base environment only needs Numpy and SimpleITK to run properly. If not already installed, run
```
pip install -r requirements.txt
```

#### nnDetection
Create a virtual environment inside the *nnDet* folder and activate it. The default environment name is **.detenv**.
```
cd nnDet
python3 -m venv .detenv
source .detenv/bin/activate
```
Then follow the instrutctions for installing nnDetection from source: [https://github.com/MIC-DKFZ/nnDetection?tab=readme-ov-file#source](https://github.com/MIC-DKFZ/nnDetection?tab=readme-ov-file#source).

**Important:** The environment variables *det_data* and *det_models* must point to *nnDet/input* and *nnDet/models* respectively! They can be set by adding the following lines to the *.detenv/bin/activate* file (or to *.bashrc*):
```
export det_data="/home/fmusio/projects/TopCoWSubmissions/nnDet/input"
export det_models="/home/fmusio/projects/TopCoWSubmissions/nnDet/model"
export OMP_NUM_THREADS=1
export det_num_threads=12
```
(Of course you need to adapt the paths. Just keep the last parts *nnDet/input* and *nnDet/model* as is!)

#### nnUNet
Create a virtual environment inside the *nnUNet* folder and activate it. The default environment name is **.segenv**.
```
cd nnUNet
python3 -m venv .segenv
source .segenv/bin/activate
```
Then follow the instrutctions for installing nnUNet: [https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions).

**Important:** The environment variables *nnUNet_raw*, *nnUNet_preprocessed* and *nnUNet_results* must point to *nnUNet/input/image*, *nnUNet/input/preprocessed* and *nnUNet/model/* respectively! They can be set by adding the following lines to the *.segenv/bin/activate* file (or to *.bashrc*):
```
export nnUNet_raw="/home/fmusio/projects/TopCoWSubmissions/nnUNet/input/image"
export nnUNet_preprocessed="/home/fmusio/projects/TopCoWSubmissions/nnUNet/input/preprocessed"
export nnUNet_results="/home/fmusio/projects/TopCoWSubmissions/nnUNet/model"
```
(Of course you need to adapt the paths. Just keep the last parts *nnUNet/input/image*, *nnUNet/input/preprocessed* and *nnUNet/model/* as is!)

### Model Weights
Since the model weights are too large for github, they can be found here: https://drive.google.com/drive/folders/14u33bdB8MawGJ7Z4M5AjNi-i3dx60yWj?usp=sharing. After cloning the repo you can download the weights from google drive and place them in the correct folders:  
For nnDet there is just 1 fold for each task. Place the saved weights for each fold inside
```
nnDet/model/<TASK NAME>/RetinaUNetV001_D3V001_3d/<FOLD>
```
For nnUNet there are 5 folds for each task (or dataset). Place the saved weights for each fold inside
```
nnUNet/model/<DATASET NAME>/nnUNetTrainer__nnUNetPlans__3d_fullres/<FOLD>
```

### Make Shell Scripts Executable
The python script *inference.py* calls the shell scripts *inference_detection.sh* and *inference_segmentation.sh* consecutively using *subprocess*. For this to work you might need to make the shell scripts executable by running
```
chmod +x inference_detection.sh inference_segmentation.sh
```

## Running Inference
Once you have placed your angiographic images in the correct input folder, you can basically just run the *inference.py* script to produce the CoW multi-class segmentations. More precisely:
1. Create the input folders *input/head-mr-angio* and *input/head-ct-angio*
2. Put your MRA images (CTA images) in *input/head-mr-angio* (*input/head-ct-angio*). 
3. Specify the track (either 'mr' or 'ct') at the top of the script *inference.py* (marked as TODO).  
    - Optionally, you can specify the name of your det_env and seg_env. Defaults are *.detenv* and *.segenv*.
4. Run the script. 
    - The outputs are stored in *output/images/cow-multiclass-segmentation*.
