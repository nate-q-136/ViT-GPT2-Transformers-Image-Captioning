import albumentations as A
from albumentations.pytorch import ToTensorV2

sample_tfms = [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
]
train_tfms = A.Compose(
    [
        *sample_tfms,
        A.Resize(224, 224),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2(),
    ]
)
val_tfms = A.Compose(
    [
        A.Resize(224, 224),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2(),
    ]
)
