import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    # ============== comp exp name =============
    comp_name = "vesuvius"

    # comp_dir_path = './'
    comp_dir_path = "/kaggle/input/"
    comp_folder_name = "vesuvius-challenge-ink-detection"
    comp_dataset_path = f"{comp_dir_path}{comp_folder_name}/"

    exp_name = "sgm_effnb0"

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = "Unet"
    backbone = [
        "mit_b2",
        "mit_b3",
        "mit_b4",
        "tu-regnety_064",
        "tu-resnest50d_4s2x40d",
        "resnet50",
        "resnet34",
    ]  # 'tu-cs3se_edgenet_x', 'se_resnext50_32x4d', 'tu-seresnextaa101d_32x8d', 'vgg19_bn', 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5', 'tu-seresnext26d_32x4d'

    in_chans = 8  # 65
    # ============== training cfg =============
    size = 224
    tile_size = 224
    stride = tile_size // 4

    batch_size = 32  # 32
    use_amp = True

    scheduler = "GradualWarmupSchedulerV2"
    # scheduler = 'CosineAnnealingLR'
    epochs = 15

    warmup_factor = 10
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 2

    objective_cv = "binary"  # 'binary', 'multiclass', 'regression'
    metric_direction = "maximize"  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = "best"  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    print_freq = 50
    num_workers = 2

    seed = 42

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf(
            [
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ],
            p=0.4,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(
            max_holes=1,
            max_width=int(size * 0.3),
            max_height=int(size * 0.3),
            mask_fill_value=0,
            p=0.5,
        ),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]
