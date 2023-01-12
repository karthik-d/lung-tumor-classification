# Lung Tumor Subtyping

- [Research Manuscript](./Documentation/Draft-1.pdf)

- Deep-learning analysis of lung histopathology images to identify lung tumor types.    
- Study of the imapct of nucleus segmentation in tumor subtyping.

![Proposed Pipeline](./figures/DrawIO/Overall-Flowchart_V3.drawio.png)

## Study Dataset

- The data of histopathology images is mainly sourced from [LC25000 Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).
  - 768 x 768 resolution images of lung histology.
  - Contains patch-level labels of tumor type.
  - HIPAA compliant and validated source.
  - [Detailed data description](https://arxiv.org/abs/1912.12142v1).
- For further analysis and segmentation, ground truth was prepared using a semi-automatic annotation strategy with the help of expert pathologists.

## Common Downstream Tumor Classification

- The classifier is a custom lightweight CNN, that performs dowstream tumor subtyping.
- The common downstream acts as a discriminator reference to compare subtyping performances with and without nucleus segmentation of histology images.

![CNN Classifier](./figures/DrawIO/Classifier-Overall_V1.drawio.png)

## Intermediate Nucleus Segmentation in the P<sub>seg</sub> Pipeline

- Nuclear regions of the histopathology images are segmented out before classification.
- **Rationale:** The nuclei portray sufficiently distinct visual characteristics under each tumor type to discern them apart.
- An Xception-style UNet architecture is trained and fine-tuned for classification.
