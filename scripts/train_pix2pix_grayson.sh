set -ex
python /home/nianyli/Desktop/code/thesis/Custom_pix2pix/train_grayson.py --dataroot ./datasets/facades --name experiment_1_3d_trans --model pix2pix --direction BtoA --crop_size 128 --netG unet_128 --use_dist_labels --dist_label_type 3D --n_epochs 1000 --n_epochs_decay 1000
