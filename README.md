# Self-Supervised Learning and CLIP/DINO

This repository contains the self-supervised learning and pretrained vision-model portions of Stanford CS231n Assignment 3. It covers SimCLR representation learning, CLIP zero-shot classification and retrieval, and DINO-based video segmentation.

## Contents

### Self-Supervised Learning with SimCLR

Notebook: [`Self_Supervised_Learning.ipynb`](Self_Supervised_Learning.ipynb)

The notebook trains and evaluates SimCLR on CIFAR-10. Two augmented views of each image form a positive pair, a ResNet encoder extracts representations, and a projection head maps them into the space used by the contrastive loss. The learned encoder is then evaluated on image classification and compared with a model trained without self-supervised pretraining.

Main implementation files:

- [`cs231n/simclr/data_utils.py`](cs231n/simclr/data_utils.py): SimCLR data augmentation and paired CIFAR-10 samples.
- [`cs231n/simclr/contrastive_loss.py`](cs231n/simclr/contrastive_loss.py): cosine similarity plus naive and vectorized SimCLR losses.
- [`cs231n/simclr/model.py`](cs231n/simclr/model.py): encoder, projection head, and downstream classifier.
- [`cs231n/simclr/utils.py`](cs231n/simclr/utils.py): training and evaluation loops.

Implementation checklist:

1. Build the random crop, horizontal flip, color jitter, and grayscale augmentation pipeline.
2. Return two independently augmented views from each CIFAR-10 sample.
3. Implement pairwise cosine similarity.
4. Implement the naive and vectorized contrastive losses.
5. Complete the SimCLR training step.
6. Train a linear classifier and compare results with and without pretrained representations.

The notebook includes numerical sanity checks for the augmentation and loss functions. Its final pretrained linear-evaluation experiment targets at least 70% top-1 accuracy.

### CLIP and DINO

Notebook: [`CLIP_DINO.ipynb`](CLIP_DINO.ipynb)

The CLIP portion uses aligned image and text embeddings for zero-shot classification and text-to-image retrieval on COCO images. The DINO portion visualizes self-attention and patch features, then trains a lightweight classifier over DINO patch embeddings for semantic video segmentation on DAVIS.

Main implementation file:

- [`cs231n/clip_dino.py`](cs231n/clip_dino.py): CLIP similarity, zero-shot classification, image retrieval, DAVIS preprocessing, IoU evaluation, and the DINO segmentation classifier.

Implementation checklist:

1. Compute vectorized cosine similarities between CLIP text and image features.
2. Implement CLIP zero-shot classification.
3. Implement top-k text-to-image retrieval.
4. Inspect DINO attention maps and visualize patch embeddings with PCA.
5. Define, train, and run inference with a lightweight DINO segmentation head.
6. Evaluate predictions with mean intersection over union and render the segmentation video.

Expected checks from the notebook:

- CLIP similarity relative error below `1e-5`.
- DINO segmentation mean IoU above `0.45` on the first test frame.
- DINO segmentation mean IoU above `0.50` on the last test frame.
- Full-video mean IoU above `0.55`.

## Running the notebooks

A CUDA-capable GPU is recommended, especially for SimCLR training and full-video DINO feature extraction. Both notebooks automatically fall back to CPU, but execution will be considerably slower.

The notebooks are configured for Google Colab. Upload the assignment directory to Google Drive, open each notebook, and update its `FOLDERNAME` variable to the directory containing this repository. Run cells from top to bottom so that dependencies, datasets, pretrained weights, and models are initialized in the expected order.

For local Jupyter use, start from this directory:

```bash
jupyter notebook
```

Install the notebook-specific packages in addition to a compatible PyTorch environment:

```bash
pip install thop ftfy regex tqdm decord scikit-learn opencv-python
pip install git+https://github.com/openai/CLIP.git
```

Data and model assets are obtained as follows:

- CIFAR-10 is downloaded automatically by `torchvision`.
- SimCLR pretrained weights are downloaded into `pretrained_model/` by the notebook.
- COCO can be downloaded with `cs231n/datasets/get_coco_dataset.sh`.
- The DINO ViT-S/8 model is loaded through `torch.hub`.
- DAVIS is downloaded automatically on the first `DavisDataset` run.

Run the notebooks in this order if you want to follow the progression from single-modality self-supervision to large pretrained vision-language and vision models:

1. `Self_Supervised_Learning.ipynb`
2. `CLIP_DINO.ipynb`
