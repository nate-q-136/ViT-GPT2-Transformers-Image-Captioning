from model.vit_gpt2_model import (
    get_vit_gpt2_model,
    get_feature_extractor,
    get_tokenizer,
)
from model.device import device
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Inference using a ViT-GPT2 model")

    parser.add_argument(
        "--model_pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--feature_extractor_path",
        type=str,
        required=True,
        help="Name of the feature extractor",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Name of the tokenizer",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image",
    )

    return parser.parse_args()
    pass


def inference(args):
    tokenizer = get_tokenizer(args.tokenizer_name)
    feature_extractor = get_feature_extractor(args.feature_extractor_path)
    model = get_vit_gpt2_model(
        feature_extractor_name=args.feature_extractor_path,
        tokenizer_name=args.tokenizer_name,
        pretrained=True,
        model_pretrained_path=args.model_pretrained_path,
    )

    model = model.to(device)

    image = Image.open(args.image_path).convert("RGB")
    generated_captions = tokenizer.batch_decode(
        model.generate(**feature_extractor(image, return_tensors="pt").to(device)),
        skip_special_tokens=True,
    )

    print("Generated Captions:")
    for caption in generated_captions:
        print(caption.replace("  ", " ").replace("  ", " "))
    pass


if __name__ == "__main__":
    args = parse_args()
    inference(args)
    pass
