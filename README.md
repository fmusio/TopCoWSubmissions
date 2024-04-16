# TopCoW Submissions
This repo contains the inference code for the [TopCoW](https://topcow23.grand-challenge.org/) multi-class CoW segmentation submissions of the organizers Fabio Musio and Kaiyuan Yang. The submissions were **ranked 2nd (MRA) and 3rd (CTA)** on our [public leaderboards](https://topcow23.grand-challenge.org/evaluation/finaltest-mra-multiclass/leaderboard/).

Details and results on our TopCoW challenge can be found in the TopCoW summary paper on arXiv: [https://arxiv.org/abs/2312.17670](https://arxiv.org/abs/2312.17670). 
```
@misc{topcow2023benchmarking,
      title={TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA}, 
      author={Kaiyuan Yang and Fabio Musio and Yihui Ma and Norman Juchler and Johannes C. Paetzold and Rami Al-Maskari and Luciano Höher and Hongwei Bran Li and Ibrahim Ethem Hamamci and Anjany Sekuboyina and Suprosanna Shit and Houjing Huang and Diana Waldmannstetter and Florian Kofler and Fernando Navarro and Martin Menten and Ivan Ezhov and Daniel Rueckert and Iris Vos and Ynte Ruigrok and Birgitta Velthuis and Hugo Kuijf and Julien Hämmerli and Catherine Wurster and Philippe Bijlenga and Laura Westphal and Jeroen Bisschop and Elisa Colombo and Hakim Baazaoui and Andrew Makmur and James Hallinan and Bene Wiestler and Jan S. Kirschke and Roland Wiest and Emmanuel Montagnon and Laurent Letourneau-Guillon and Adrian Galdran and Francesco Galati and Daniele Falcetta and Maria A. Zuluaga and Chaolong Lin and Haoran Zhao and Zehan Zhang and Sinyoung Ra and Jongyun Hwang and Hyunjin Park and Junqiang Chen and Marek Wodzinski and Henning Müller and Pengcheng Shi and Wei Liu and Ting Ma and Cansu Yalçin and Rachika E. Hamadache and Joaquim Salvi and Xavier Llado and Uma Maria Lal-Trehan Estrada and Valeriia Abramova and Luca Giancardo and Arnau Oliver and Jialu Liu and Haibin Huang and Yue Cui and Zehang Lin and Yusheng Liu and Shunzhi Zhu and Tatsat R. Patel and Vincent M. Tutino and Maysam Orouskhani and Huayu Wang and Mahmud Mossa-Basha and Chengcheng Zhu and Maximilian R. Rokuss and Yannick Kirchhoff and Nico Disch and Julius Holzschuh and Fabian Isensee and Klaus Maier-Hein and Yuki Sato and Sven Hirsch and Susanne Wegener and Bjoern Menze},
      year={2023},
      eprint={2312.17670},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Introduction
The Circle of Willis (CoW) is an important anastomotic network of arteries connecting the anterior and posterior circulations of the brain, as well as the left and right cerebral hemispheres.  Due to its centrality, the CoW is commonly involved in pathologies like aneurysms and stroke. Hence, an accurate characterization of the CoW is of great clinical relevance. 

The TopCoW challenge was the first public challenge on CoW angio-architecture extraction and brain vessel segment annotation on two common angiographic imaging modalities, namely magnetic resonance angiography (MRA) and computed tomography angiography (CTA). 
The aim of the challenge was to extract the CoW vessels from 3D angiographic imaging by segmentation of the vessel components. For that purpose, we released a new [dataset](https://topcow23.grand-challenge.org/data/) of joint-modalities, CTA and MRA of the same patient cohort, both with annotations of the anatomy of the CoW.

As organizers, we submitted baseline algorithms for the CoW multi-class segmentation for both MRA and CTA. The inference code and model weights are published in this repo and can be used freely for CoW vessel segmentation tasks.

## Method Description
A simple two-stage approach was employed: The [nnDetection](https://github.com/MIC-DKFZ/nnDetection) framework was used to detect and extract custom ROIs containing the CoW based on the binary labels and a 3D [nnUNet](https://github.com/MIC-DKFZ/nnUNet) was employed for the subsequent multi-class segmentation on the ROIs. Additionally, inter-modal registration was used as a data augmentation strategy, registering all the image pairs and thereby doubling the size of the training set for both modalities.

## Usage
### Prerequisites
Both the nnDetection and the nnUNet frameworks need to be installed. Due to a lack of compatibility of their respective dependencies, two separate environments must be installed and activated consecutively. 

Other than that, the base environment only needs Numpy and SimpleITK to run properly. If not already installed, run
```
pip install -r requirements.txt
```

#### nnDetection
Create a virtual environment inside the *nnDet* folder and activate it. The default name is **.detenv**.
```
cd nnDet
python3 -m venv .detenv
source .detenv/bin/activate
```
Then follow the instrutctions for installing nnDetection from source: [https://github.com/MIC-DKFZ/nnDetection?tab=readme-ov-file#source](https://github.com/MIC-DKFZ/nnDetection?tab=readme-ov-file#source).

**Important:** The environment variables *det_data* and *det_models* must point to *nnDet/input* and *nnDet/models* respectively! They can be set by adding the following lines to the *.detenv/bin/activate* file (or to *.bashrc*):
```
export det_data="/home/fmusio/projects/topcowsubmission/BaselineAlgoMRMulticlass/nnDet/input"
export det_models="/home/fmusio/projects/topcowsubmission/BaselineAlgoMRMulticlass/nnDet/model"
export OMP_NUM_THREADS=1
```
(Of course you need to adapt the paths. Just keep the last part *nnDet/input* and *nnDet/model* as is!)

#### nnUNet
Create a virtual environment inside the *nnUNet* folder and activate it. The default name is **.segenv**.
```
cd nnUNet
python3 -m venv .segenv
source .segenv/bin/activate
```
Then follow the instrutctions for installing nnUNet from source: [https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md#installation-instructions).

**Important:** The environment variables *nnUNet_raw*, *nnUNet_preprocessed* and *nnUNet_results* must point to *nnUNet/input/image*, *nnUNet/input/preprocessed* and *nnUNet/model/* respectively! They can be set by adding the following lines to the *.segenv/bin/activate* file (or to *.bashrc*):
```
export nnUNet_raw="/home/fmusio/projects/topcowsubmission/BaselineAlgoMRMulticlass/nnUNet/input/image"
export nnUNet_preprocessed="/home/fmusio/projects/topcowsubmission/BaselineAlgoMRMulticlass/nnUNet/input/preprocessed"
export nnUNet_results="/home/fmusio/projects/topcowsubmission/BaselineAlgoMRMulticlass/nnUNet/model"
```
(Of course you need to adapt the paths. Just keep the last parts *nnUNet/input/image*, *nnUNet/input/preprocessed* and *nnUNet/model/* as is!)

## Running Inference
Once you have placed your angiographic images in the correct input folder, you can basically just run the *inference.py* script to produce the CoW multi-class segmentation. More precisely:
1. Store your MRA images (CTA images) in *input/head-mr-angio* (*input/head-ct-angio*)
2. Specify the track (either 'mr' or 'ct') at the top of the script *inference.py*.  
    - Optionally, you can specify the name of your det_env and seg_env. Defaults are *.detenv* and *.segenv*
3. Run the script. 
    - The outputs are stored in *output/images/cow-multiclass-segmentation*









