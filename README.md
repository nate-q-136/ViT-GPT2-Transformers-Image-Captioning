
# ViT-GPT2-Transformers-Image-Captioning

This project implements a Vision Transformer (ViT) and GPT-2-based model for image captioning on datasets such as COCO, Flickr8k, and Flickr30k. The model uses the pre-trained Vision Transformer (ViT) to extract image features and GPT-2 to generate captions based on these features.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [COCO Dataset](#train-on-coco-dataset)
  - [Flickr8k Dataset](#train-on-flickr8k-dataset)
  - [Flickr30k Dataset](#train-on-flickr30k-dataset)
- [Model Inference](#model-inference)
- [Weights and Biases Integration](#weights-and-biases-integration)
- [References](#references)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your_username/ViT-GPT2-Transformers-Image-Captioning.git
cd ViT-GPT2-Transformers-Image-Captioning
```

### 2. Create a virtual environment (optional)
It is recommended to create a virtual environment to manage the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
Install the required Python libraries from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `transformers`
- `torch`
- `albumentations`
- `firebase-admin`
- `argparse`
- `rouge-score`

Make sure you have installed all these packages.

### 4. Install additional tools
- Install **CUDA** if you want to run the model on a GPU for faster training.
- Install **Weights and Biases** (WandB) if you want to log your training and evaluation metrics.

## Dataset Preparation

### 1. COCO Dataset
Download the COCO dataset from the official website:

- COCO dataset: [COCO 2017 Train/Val Images](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

Extract the files and organize the directory structure as follows:

```
COCO/
├── train2017/
│   ├── ... (JPEG images)
├── val2017/
│   ├── ... (JPEG images)
└── annotations/
    ├── captions_train2014.json
    ├── captions_val2014.json
```

### 2. Flickr8k Dataset
Download the Flickr8k dataset:
- [Flickr8k Images](https://www.kaggle.com/datasets/adityajn105/flickr8k?select=Images)

Unzip the files and place them in an appropriate directory:

```
Flickr8k/
├── images/
│   ├── ... (JPEG images)
└── captions.txt
```

### 3. Flickr30k Dataset
Download the Flickr30k dataset:
- [Flickr30k Dataset](https://www.kaggle.com/datasets/eeshawn/flickr30k)

Organize the directory as follows:

```
Flickr30k/
├── flickr30k_images/
│   ├── ... (JPEG images)
└── captions.txt
```

## Training

You can train the ViT-GPT2 model on different datasets (COCO, Flickr8k, Flickr30k) by running the corresponding training scripts. In this repository, I prefer to use feature extraction from the "google/vit-base-patch16-224" and tokenizer "gpt2".

### 1. Train on COCO Dataset

To train the model on the COCO dataset, run the `train_coco.py` script:

```bash
python src/train_coco.py \
    --tokenizer_name <TOKENIZER_NAME> \
    --feature_extractor_name <FEATURE_EXTRACTOR_NAME> \
    --coco_data_folder_path <COCO_DATA_FOLDER_PATH> \
    --folder_images <FOLDER_IMAGES> \
    --file_caption_json_name <CAPTION_JSON_NAME> \
    --epochs 5 \
    --batch_size 2 \
    --logging_steps 10 \
    --save_steps 10 \
    --warmup_steps 10 \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --output_dir ViT_GPT2_Coco \
    --wandb_project coco-vit-gpt-experiment
```

### 2. Train on Flickr8k Dataset

To train the model on the Flickr8k dataset, use the `train_flickr8k.py` script:

```bash
python src/train_flickr8k.py \
    --tokenizer_name <TOKENIZER_NAME> \
    --feature_extractor_name <FEATURE_EXTRACTOR_NAME> \
    --flickr8k_data_folder_path <FLICKR8K_DATA_FOLDER_PATH> \
    --folder_images <FOLDER_IMAGES> \
    --file_caption_txt <CAPTION_TXT_NAME> \
    --epochs 5 \
    --batch_size 2 \
    --logging_steps 10 \
    --save_steps 10 \
    --warmup_steps 10 \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --output_dir ViT_GPT2_Flickr8k \
    --wandb_project flickr8k-vit-gpt-experiment
```

### 3. Train on Flickr30k Dataset

For the Flickr30k dataset, use the `train_flickr30k.py` script:

```bash
python src/train_flickr30k.py \
    --tokenizer_name <TOKENIZER_NAME> \
    --feature_extractor_name <FEATURE_EXTRACTOR_NAME> \
    --flickr30k_data_folder_path <FLICKR30K_DATA_FOLDER_PATH> \
    --folder_images <FOLDER_IMAGES> \
    --file_caption_txt <CAPTION_TXT_NAME> \
    --epochs 5 \
    --batch_size 2 \
    --logging_steps 10 \
    --save_steps 10 \
    --warmup_steps 10 \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --output_dir ViT_GPT2_Flickr30k \
    --wandb_project flickr30k-vit-gpt-experiment
```

If you want to train your pretrained model, you can pass some arguments to the script:
```bash
python src/train_flickr30k.py \
    --is_pretrained \
    --pretrained_model_path <PRETRAINED_MODEL_PATH> \
    --tokenizer_name <TOKENIZER_NAME> \
    --feature_extractor_name <FEATURE_EXTRACTOR_NAME> \
    --flickr30k_data_folder_path <FLICKR30K_DATA_FOLDER_PATH> \
    --folder_images <FOLDER_IMAGES> \
    --file_caption_txt <CAPTION_TXT_NAME> \
    --epochs 5 \
    --batch_size 2 \
    --logging_steps 10 \
    --save_steps 10 \
    --warmup_steps 10 \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --output_dir ViT_GPT2_Flickr30k \
    --wandb_project flickr30k-vit-gpt-experiment

```

## Model Inference

Once the model is trained, you can perform inference using the following script:

```bash
python src/inference.py \
    --model_path <MODEL_PATH> \
    --tokenizer_name <TOKENIZER_NAME> \
    --feature_extractor_name <FEATURE_EXTRACTOR_NAME> \
    --image_path <IMAGE_PATH>
```

### Example:

```bash
python src/inference.py \
    --model_path ViT_GPT2_Coco/checkpoint-1000 \
    --tokenizer_name gpt2 \
    --feature_extractor_name google/vit-base-patch16-224 \
    --image_path path_to_image.jpg
```

## Weights and Biases Integration

This project supports integration with Weights and Biases (WandB) for logging and tracking experiments. To enable WandB logging, simply specify the `--wandb_project` argument in the training scripts.

Ensure that you are logged into your WandB account using:
```bash
wandb login
```

## References

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [GPT-2](https://openai.com/research/gpt-2)
- [COCO Dataset](https://cocodataset.org/)
- [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398)
- [Flickr30k Dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset)
