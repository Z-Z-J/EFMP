# EFMP: Extrinsic Parameters-free Multi-view 3D human pose estimation

This repository is the official implementation of [EFMP: Extrinsic Parameters-free Multi-view 3D human pose estimation]. 

## Key Idea: Data pre-processing + SVJFormer + BPMA

## Requirements

To install requirements:

```setup
#1. Create a conda virtual environment.
conda create -n mvhpe python=3.7.11
conda activate mvhpe

#2. Install Pytorch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#3. Install requirements.
pip install -r requirements.txt
```
## Preparing Datasets and Rre-trained model
1. Download the required data.
   * Download our data from [Google Drive](https://drive.google.com/drive/folders/1Z6-fLuANi2Y67w-VZrx-oG_K9IrSINtK?usp=sharing) 
   * Download our pretrained model from [Google Drive](https://drive.google.com/drive/folders/10YgOngKAVqAjWuplS8fD0ZwowhVz5Lgr?usp=drive_link)
   
2. You need to add the `data` and  `checkpoint` as below.
```
|-- dataset
`-- |-- h36m_sub1.npz
    `-- ...
    `-- h36m_sub11.npz
|-- checkpoint
`-- |-- h36m_cpn_wo_parameters.pth
    `-- h36m_cpn_w_intrinsic_parameters.pth
    `-- h36m_gt_w_intrinisc_parameters.pth
```


## Training

To train our model, run:

```train
python run_h36m_cpn_wo_parameters.py --train --frame 1 - 
python run_h36m_cpn_w_intrinsic_parameters.py --train --frame 1  
python run_h36m_gt_w_intrinsic_parameters.py --train --frame 1  
```

## Evaluation

To evaluate our model, run:

```eval
python run_h36m_cpn_wo_parameters.py --test --frame 1 --resume --previous_dir ./checkpoint/h36m_cpn_wo_parameters.pth 
python run_h36m_cpn_w_intrinsic_parameters.py --test --frame 1 --resume --previous_dir ./checkpoint/h36m_cpn_w_intrinsic_parameters.pth 
python run_h36m_gt_w_intrinsic_parameters.py --test --frame 1 --resume --previous_dir ./checkpoint/h36m_gt_w_intrinsic_parameters.pth 
```
## Results

Our model achieves the following performance on Human3.6M:

| Methods            |Camera     |MPJPE|
| -------------------|-----------|------------|
| Ours (CPN)   |camera-parameters-free|     25.8mm |      
| Ours (CPN)  |camera-intrinsic-parameters-free|     25.0mm  |  
| Ours (GT)  |camera-intrinsic-parameters-free|     5.6mm |  

