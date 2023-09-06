# We present instructions example of running baselines, you can adjust by yourself.

# CIFAR10

# DLG
python rec_mult.py --optim zhu --save_image --cost_fn l2  --indices def  --weights equal --init randn --max_iterations 500 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 1  --lr 1e-1
python rec_mult.py --optim zhu --save_image --cost_fn l2  --indices def  --weights equal --init randn --max_iterations 500 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 4  --lr 1e-1
python rec_mult.py --optim zhu --save_image --cost_fn l2  --indices def  --weights equal --init randn --max_iterations 500 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 8  --lr 1e-1

# IG
python rec_mult.py --signed --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 10000 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 1  --lr 1e-1 --tv 1e-6
python rec_mult.py --signed --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 10000 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 4  --lr 1e-1 --tv 1e-6
python rec_mult.py --signed --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 10000 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 8  --lr 1e-1 --tv 1e-6

# GIAS
python  rec_mult.py  --optim gias --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 3000 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 1  --gias --lr 1e-1 --tv 1e-6 --generative_model DCGAN --gen_dataset CIFAR10 --gias_lr 1e-4 --gias_iterations 3000
python  rec_mult.py  --optim gias --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 3000 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 4  --gias --lr 1e-1 --tv 1e-6 --generative_model DCGAN --gen_dataset CIFAR10 --gias_lr 1e-4 --gias_iterations 3000
python  rec_mult.py  --optim gias --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 3000 --model ResNet18 --dataset CIFAR10 --data_path /home/ubuntu/data/CIFAR10 --num_images 8  --gias --lr 1e-1 --tv 1e-6 --generative_model DCGAN --gen_dataset CIFAR10 --gias_lr 1e-4 --gias_iterations 3000

# ImageNet

# DLG

# DLG
python rec_mult.py --optim zhu --save_image --cost_fn l2  --indices def  --weights equal --init randn --max_iterations 1000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet --num_images 1  --lr 1e-1
python rec_mult.py --optim zhu --save_image --cost_fn l2  --indices def  --weights equal --init randn --max_iterations 1000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet --num_images 4  --lr 1e-1
python rec_mult.py --optim zhu --save_image --cost_fn l2  --indices def  --weights equal --init randn --max_iterations 1000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet --num_images 8  --lr 1e-1

# IG
python rec_mult.py --signed --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 48000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet --num_images 1  --lr 1e-1 --tv 1e-4
python rec_mult.py --signed --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 48000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet --num_images 4  --lr 1e-1 --tv 1e-4
python rec_mult.py --signed --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 48000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet --num_images 8  --lr 1e-1 --tv 1e-4

# GIAS
python  rec_mult.py  --optim gias --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 3000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet/ --num_images 1 --gias --lr 1e-1 --tv 1e-4 --generative_model stylegan2-ada --gen_dataset I64 --gias_lr 1e-4 --gias_iterations 3000
python  rec_mult.py  --optim gias --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 3000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet/ --num_images 4 --gias --lr 1e-1 --tv 1e-4 --generative_model stylegan2-ada --gen_dataset I64 --gias_lr 1e-4 --gias_iterations 3000
python  rec_mult.py  --optim gias --save_image --cost_fn sim  --indices def  --weights equal --init randn --max_iterations 3000 --model ResNet18 --dataset I64 --data_path /home/ubuntu/data/ImageNet/ --num_images 8 --gias --lr 1e-1 --tv 1e-4 --generative_model stylegan2-ada --gen_dataset I64 --gias_lr 1e-4 --gias_iterations 3000
