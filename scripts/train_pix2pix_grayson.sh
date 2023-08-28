set -ex
python train_grayson.py --dataroot ./datasets/facades --name experiment2 --model pix2pix --direction BtoA --crop_size 128 --netG unet_128 --use_dist_labels
