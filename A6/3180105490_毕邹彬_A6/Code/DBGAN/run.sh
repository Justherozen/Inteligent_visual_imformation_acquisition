conda activate DBGAN

mkdir pro_data_blurred

python gauss_blur.py

# python motion_blur/blur_image.py

python datasets/split_train_test_val.py --fold_blurred pro_data_blurred --fold_sharp pro_data --output pro_data_AB

python datasets/combine_A_and_B.py --fold_A pro_data_AB/blurred --fold_B pro_data_AB/sharp --fold_AB pro_data_AB/combined

python -m visdom.server

python train.py --dataroot pro_data_AB/combined --learn_residual --resize_or_crop crop --fineSize 256

cp -r pro_data_AB/blurred pro_data_AB/generated

python test.py --dataroot pro_data_AB/generated --model test --dataset_mode single --learn_residual