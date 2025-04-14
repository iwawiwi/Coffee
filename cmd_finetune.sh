#!bin/bash
# Original USKC, no RandAug, no SE, constant lr, max ep 10, lr 1e-4
python train.py --dataset uskc --rand_aug False --add_se False --num_epochs 10 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_no-randaug_no-se_cosine-lr_max-ep-10_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Original USKC, no RandAug, no SE, constant lr, max ep 20, lr 1e-4
python train.py --dataset uskc --rand_aug False --add_se False --num_epochs 20 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_no-randaug_no-se_cosine-lr_max-ep-20_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Defect USKC, no RandAug, no SE, constant lr, max ep 10, lr 1e-4
# python train.py --dataset uskc_defect --rand_aug False --add_se False --num_epochs 10 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save defuskc_r18_no-randaug_no-se_cosine-lr_max-ep-10_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Defect USKC, no RandAug, no SE, constant lr, max ep 20, lr 1e-4
# python train.py --dataset uskc_defect --rand_aug False --add_se False --num_epochs 20 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save defuskc_r18_no-randaug_no-se_cosine-lr_max-ep-20_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&

# Original USKC, RandAug, no SE, constant lr, max ep 10, lr 1e-4
python train.py --dataset uskc --rand_aug True --add_se False --num_epochs 10 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_randaug_no-se_cosine-lr_max-ep-10_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Original USKC, RandAug, no SE, constant lr, max ep 20, lr 1e-4
python train.py --dataset uskc --rand_aug True --add_se False --num_epochs 20 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_randaug_no-se_cosine-lr_max-ep-20_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Defect USKC, RandAug, no SE, constant lr, max ep 10, lr 1e-4
# python train.py --dataset uskc_defect --rand_aug True --add_se False --num_epochs 10 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save defuskc_r18_randaug_no-se_cosine-lr_max-ep-10_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Defect USKC, RandAug, no SE, constant lr, max ep 20, lr 1e-4
# python train.py --dataset uskc_defect --rand_aug True --add_se False --num_epochs 20 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save defuskc_r18_randaug_no-se_cosine-lr_max-ep-20_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&

# Original USKC, no RandAug, SE, constant lr, max ep 10, lr 1e-4
python train.py --dataset uskc --rand_aug False --add_se True --num_epochs 10 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_no-randaug_se_cosine-lr_max-ep-10_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Original USKC, no RandAug, SE, constant lr, max ep 20, lr 1e-4
python train.py --dataset uskc --rand_aug False --add_se True --num_epochs 20 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_no-randaug_se_cosine-lr_max-ep-20_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Defect USKC, no RandAug, SE, constant lr, max ep 10, lr 1e-4
# python train.py --dataset uskc_defect --rand_aug False --add_se True --num_epochs 10 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save defuskc_r18_no-randaug_se_cosine-lr_max-ep-10_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Defect USKC, no RandAug, SE, constant lr, max ep 20, lr 1e-4
# python train.py --dataset uskc_defect --rand_aug False --add_se True --num_epochs 20 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save defuskc_r18_no-randaug_se_cosine-lr_max-ep-20_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&

# Original USKC, RandAug, SE, constant lr, max ep 10, lr 1e-4
python train.py --dataset uskc --rand_aug True --add_se True --num_epochs 10 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_randaug_se_cosine-lr_max-ep-10_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee &&
# Original USKC, RandAug, SE, constant lr, max ep 20, lr 1e-4
python train.py --dataset uskc --rand_aug True --add_se True --num_epochs 20 --lr 1e-4 --model resnet18 --scheduler cosine --ckpt_save uskc_r18_randaug_se_cosine-lr_max-ep-20_lr-1e-4.pth --root_path ~/Documents/research/dataset/USK_coffee
