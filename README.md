# Lung Tumor Subtyping from Histology and the Impact of Nucleus Segmentation on Tumor Discernibility

## Cite Us

[Link to the Research Paper](https://link.springer.com/article/10.1007/s13721-023-00417-2).

If you find our work useful in your research, please cite us:

```
@article{jsm2023,
  title={A deep learning approach for nucleus segmentation and tumor classification from lung histopathological images},
  author={Jaisakthi, SM and Desingu, Karthik and Mirunalini, P and Pavya, S and Priyadharshini, N},
  journal={Network Modeling Analysis in Health Informatics and Bioinformatics},
  volume={12},
  number={1},
  pages={22},
  year={2023},
  publisher={Springer},
  url={https://link.springer.com/article/10.1007/s13721-023-00417-2},
  doi={https://doi.org/10.1007/s13721-023-00417-2}
}
```

### Brief Overview

<img src="./figures/to-publish/Overall-Flowchart_V3.jpg" width="750">

- Deep-learning analysis of lung histopathology images to discern lung tumor types.    
- Study of the **impact of nucleus segmentation** on tumor subtyping.
- The study method is summarized as a **brief algorithm** below.    
      
  <img src="./figures/Study-Method.png" width="400">

## Study Dataset and Data Processing

- The data of histopathology images is mainly sourced from [LC25000 Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).
  - *768 x 768* resolution images of lung histology.
  - Contains patch-level labels of tumor type.
  - [HIPAA](https://www.cdc.gov/phlp/publications/topic/hipaa.html) compliant and validated source.
  - [Detailed data description](https://arxiv.org/abs/1912.12142v1).
  
- For **further analysis and segmentation**, ground truth was prepared using a **semi-automatic annotation strategy** with the help of expert pathologists.
  - Automated nuclear region annotation is obtained using a **stain-based color thresholding approach**, detailed in the algorithm below.   
        
    <img src="./figures/Annotation-Algorithm.png" width="400" />     
  - The obtained annotations are corrected and **validated by expert pathologists**.
  - Multiple pathologist corrections are compared and averaged. The comparison is assessed using an inter-rater agreement score, namely **generalized conformity index (GCI)**; a GCI score of 0.89 was obtained.

## Common Downstream Tumor Classification

- The classifier is a **custom lightweight Convolution Neural Network**, that performs downstream tumor subtyping.
- The common downstream serves the role of a discriminator reference to compare subtyping performances with and without prior nucleus segmentation of histology images.
   
  <img src="./figures/to-publish/Classifier-Overall_V2.jpg" width="600">

## Intermediate Nucleus Segmentation in the P<sub>seg</sub> Pipeline

- **Rationale:** The nuclei portray sufficiently distinct visual characteristics under each tumor type to discern them apart.
- Nuclear regions of the lung histology images are segmented out before classification.
- An segmentation arhictecture derived heavily from the [Xception-style UNet](https://keras.io/examples/vision/oxford_pets_image_segmentation/) is trained and fine-tuned to automate this nucleus segmentation.
