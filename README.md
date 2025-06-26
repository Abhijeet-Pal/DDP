# DDP Project on the topic "Handling imbalance and few-sample size in ML based Onion disease classification"


**Description:** We propose a CNN based model for Onion Disease classification which mitigates data imbalance among the diseases classes, as there is prevalance of some diseases more than the others. 
Different CNN models are used as feature extractors, loss functions, data imbalance mitigation techniques and augmentations techniques has been compared.


# Different Techniques Used are : 
## CNN Models 
- ResNet50
- DenseNet121
## Imbalance Mitigation Techniques
- Weighted Cross Entropy Loss
- Imbalanced Dataset Sampler
- Focal Loss
## Augmentation Techniques
- Albumentations based Augmentations
- Cut-Mix Augmentation

## Requirements
The code files and models have used 
- Python Version : 3.8.10
- Cuda Version : 11.4

For Diffusion model training to generate new images, i.e. for Generate_Images_Using_Diffusion.ipynb file:
- Python Version : 3.11.11
- Cuda Version : 11.8  
Note that Generate_Images_Using_Diffusion.ipynb does not work with python version 3.8.10, and cuda version 11.4

## Dataset Structure : 
In the ipynb files, we expect that the Folder should consist of Combined Files and class wise folders with each folder containing the corresponding .jpg or .png files of the onion pest images  
**Structure:**
Combined_Files/  
├── Anthracnose/  
│ ├── anthra1.jpg  
│ ├── anthra2.png  
│ └── ...  
├── Thrips/  
│ ├── thrips1.jpg  
│ ├── thrips2.png  
│ └── ...  
├── Twister/  
│ ├── twister1.jpg  
│ ├── twister2.png  
│ └── ...  
├── Purple Blotch/  
│ ├── purpleblotch1.jpg  
│ ├── purpleblotch2.png  
│ └── ...  
├── Healthy/  
│ ├── healthy1.jpg  
│ ├── healthy2.png  
│ └── ...  
└── code_file.ipynb  

# IPYNB file Descriptions:
##

## Comparison_with_YOLO-ODD_Densenet121_CBAM_CutMix_WCE_one_subfolder.ipynb 
Here, we have compared our model of DenseNet121 with CBAM, weighted cross entropy loss, cutmix augmentations with the YOLO-ODD model using only one subfolder of the Raw Dataset, with nearly 200 images each for one class. There are only 5 classes in this subfolder of the dataset. 
## Comparison_with_YOLO-ODD_Densenet121_CBAM_CutMix_WCE_with_all_Images
 Here, we have compared our model of DenseNet121 with CBAM, weighted cross entropy loss, cutmix augmentations with the YOLO-ODD model using all the images of the 5 specific classes which are used in YOLO-ODD. Our dataset is different compared to YOLO-ODD. Also, there are about 193 images which are present in Twister as well as Anthracnose Classes.

## Comparison_with_YOLO-ODD_Densenet121_CBAM_no_augmentation_WCE_one_subfolder
Here, we have compared our model of DenseNet121 with CBAM, weighted cross entropy loss, no augmentations with the YOLO-ODD model using only one subfolder of the Raw Dataset, with nearly 200 images each for one class. There are only 5 classes in this subfolder of the dataset. 

## Densenet121_CBAM_Albumentations_Cutmix_WCE.ipynb
Here, we have used Densenet121 model, with CBAM and Weighted Cross Entropy Loss, albumentations, cutmix. For 8 classes, Anthracnose and Twister are combined. Dataset structure needs to be same. Have combined the classes in the initial code blocks, don't need to make a new dataset. 


## DenseNet121_CBAM_CutMix_WCE.ipynb
Here, we have used DenseNet121 with CBAM, WCE loss and CutMix augmentations for 8 classes, twister and anthracnose combined. This is our best model with 96.9% accuracy

## DenseNet121_CBAM_Focal_Loss.ipynb
We use DenseNet121 with CBAM, Focal Loss, no augmentations for 8 classes, twister and anthracnose combined.

## DenseNet121_CBAM_WCE.ipynb
Have used DenseNet121 with CBAM, WCE, no augmentations for 8 classes, twister and anthracnose combined.

## DenseNet121_LORA_WCE.ipynb 
We use LORA (Low Rank Adaptation) to train the initial layer of a DenseNet121 model. Need dataset structure to be as shown in Dataset Structure above, no other changes required 



## DenseNet121_Margin_Loss
Have experimented with basic DenseNet121 model and Margin Loss.

## DenseNet121_Twister_Dropped_and_Anthracnose_Twister_Merged_experiments.ipynb
Here, we have combined Anthracnose and Twister classes (7 disease + 1 Healthy), as well as Dropped Twister (7 disease + 1 Healthy) to check if Anthracnose and Twister should be considered the same, these experiments are in this ipynb file. Need only the dataset structure to be the same. No other changes required.


## DenseNet121_WCE_Albumentations.ipynb
Have used DenseNet121 with WCE loss and Albumentations based augmentations for 8 classes, anthracnose and twister combined.

## DenseNet121_WCE_CutMix.ipynb
Have used DenseNet121 with WCE loss and Cutmix augmentations for 8 classes, anthracnose and twister combined.

## Generate_Images_Using_Diffusion.ipynb
Here we generate images of the pest crops using diffusion models. Requires Python Version : 3.11.11 and Cuda Version : 11.8. Need to run it for more epochs.


## ResNet50_DenseNet121_Imbalanced_Sampler.ipynb
Here, we have defined ResNet50 and DenseNet121 model, with Weighted Cross Entropy loss as well as Imbalanced Dataset Sampler. Dataset Structure needs to be the same, other than that no other changes required. 

## ResNet50_DenseNet121_WCE_9_Class.ipynb
Here, we have experiments considering all the 9 classes (8 Disease + 1 Healthy), with both ResNet50 and DenseNet121 models. Dataset structure should be same, no other changes required.

## ViT_Experiments.ipynb
We also try to classify the images using Vision Transformers, however as the dataset does not have many images, accuracy is low.

# Best Model
Best model is DenseNet121 with CBAM, Cutmix and WCE loss. File : DenseNet121_CBAM_CutMix_WCE
