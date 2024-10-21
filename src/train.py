import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator
from model.vit_gpt2_model import get_feature_extractor, get_tokenizer, get_vit_gpt2_model
from dataset.coco_dataset import ViTGPT2Dataset
from dataset.utils import CocoUtils
from rouge_score import rouge_scorer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from model.device import device



def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT-GPT2 model on COCO dataset")
    parser.add_argument('--tokenizer_name', type=str, required=True, help='Name of the tokenizer')
    parser.add_argument('--feature_extractor_name', type=str, required=True, help='Name of the feature extractor')
    parser.add_argument('--coco_data_folder_path', type=str, required=True, help='Path to the COCO dataset folder')
    parser.add_argument('--file_caption_json_name', type=str, required=True, help='Name of the JSON file with captions')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation dataset size as a fraction')
    parser.add_argument('--is_pretrained', action='store_true', help='Whether to use a pretrained model')
    parser.add_argument('--model_pretrained_name', type=str, default=None, help='Name of the pretrained model to use')

    return parser.parse_args()

def train(args=None):
    print("Starting training...")
    tokenizer = get_tokenizer(name=args.tokenizer_name)
    print("Tokenizer loaded.")
    feature_extractor = get_feature_extractor(name=args.feature_extractor_name)
    print("Feature extractor loaded.")
    model = get_vit_gpt2_model(feature_extractor_name=args.feature_extractor_name, tokenizer_name=args.tokenizer_name, pretrained=args.is_pretrained, model_pretrained_name=args.model_pretrained_name).to(device)
    print("Model loaded.")
    sample_tfms = [
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.ColorJitter(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
        A.HueSaturationValue(p=0.3),
    ]
    train_tfms = A.Compose([
        *sample_tfms,
        A.Resize(224, 224),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2()
    ])
    val_tfms = A.Compose([
        A.Resize(224, 224),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2()
    ])
    print("Data augmentation and preprocessing loaded.")
    coco_utils = CocoUtils(root_folder_path=args.coco_data_folder_path)
    df = coco_utils.convert_to_dataframe(file_caption_json_name=args.file_caption_json_name)
    train_df, val_df = coco_utils.get_train_val_df(df=df, val_size=args.val_size)
    train_dataset = ViTGPT2Dataset(df=train_df, tokenizer=tokenizer, feature_extractor=feature_extractor, transform=train_tfms)
    val_dataset = ViTGPT2Dataset(df=val_df, tokenizer=tokenizer, feature_extractor=feature_extractor, transform=val_tfms)
    print("Datasets loaded.")
    def compute_metrics(pred):
        # Lấy các label (ground truth) và các prediction từ đầu ra của mô hình
        labels_ids = pred.label_ids  # default key label of Trainer Transformer
        pred_ids = pred.predictions  # default key output of Trainer Transformer

        # Loại bỏ các token không cần thiết
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # Tạo scorer để tính điểm ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)

        # Tính ROUGE cho từng cặp prediction và label
        rouge2_precision = 0
        rouge2_recall = 0
        rouge2_fmeasure = 0
        num_examples = len(pred_str)

        for pred, label in zip(pred_str, label_str):
            rouge_scores = scorer.score(pred, label)
            rouge2_precision += rouge_scores['rouge2'].precision
            rouge2_recall += rouge_scores['rouge2'].recall
            rouge2_fmeasure += rouge_scores['rouge2'].fmeasure

        # Tính trung bình
        rouge2_precision /= num_examples
        rouge2_recall /= num_examples
        rouge2_fmeasure /= num_examples

        # Trả về các giá trị precision, recall và F1
        return {
            "rouge2_precision": round(rouge2_precision, 4),
            "rouge2_recall": round(rouge2_recall, 4),
            "rouge2_fmeasure": round(rouge2_fmeasure, 4),
        }
    
    print("Metrics computed.")
    training_args = Seq2SeqTrainingArguments(
        output_dir="COCO_ViT_GPT2",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        evaluation_strategy="steps",
        do_train=True,
        do_eval=True,
        do_predict=True,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=10,
        warmup_steps=10,
        num_train_epochs=5,
        save_total_limit=3,
        dataloader_num_workers=1,
        report_to="wandb",
        run_name="coco-vit-gpt-experiment"
    )
    print("Training arguments loaded.") 

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )
    print("Trainer loaded.")
    print("Starting training...")

    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    train(args)
    pass
