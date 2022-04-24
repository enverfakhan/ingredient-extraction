#!/bin/bash
#
# CompecTA (c) 2018
#
# TORCH job submission script
#
# TODO:
#   - Set name of the job below changing "Keras" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch keras_pulsar_submit.sh
#
# -= Resources =-
#
# akya-cuda barbun-cuda single(1 core 15 gün) 
# short(4 saat) mid1(4 gün) mid2(8 gün) long(15gün) debug
#SBATCH -p palamut-cuda
#SBATCH -A eurocc4
#SBATCH -J tagging
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16 ### gpu*10
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --time=3-00:00:00
####SBATCH --time=0-00:15:00
##SBATCH --workdir=/truba/home/efakhan/foodstyles/ingredient-extraction
#SBATCH -o /truba/home/efakhan/foodstyles/ingredient-extraction/slurm_outs/out-%j.out  # send stdout to outfile
#SBATCH -e /truba/home/efakhan/foodstyles/ingredient-extraction/slurm_outs/err-%j.err  # send stderr to errfile

## module load centos7.9/lib/cuda/11.4
## module load centos7.3/comp/gcc/6.4
##'blocks.2.0.bn3.bias','blocks.4.0.bn3.bias','blocks.3.0.bn3.bias','blocks.5.0.bn3.bias' 

###################  Bu arayi degistirmeyin ##########################
# export PATH=${PATH}:/truba/sw/centos7.3/lib/cuda/10.1/bin/
# export LD_LIBRARY_PATH=/truba/home/akindiroglu/Workspace/Libs/cuda/lib64/
######################################################################


module purge #Olası hataları önlemek için bütün ortam modüllerini temizleyin
eval "$(/truba/home/efakhan/miniconda3/bin/conda shell.bash hook)" #Conda komutlarını aktif hale getirin
conda activate tf-gpu #Yarattığınız conda ortamını aktive edin
module load centos7.9/lib/cuda/11.4 #CUDA modülünü yükleyin

## /truba/home/akindiroglu/Workspace/Libs/miniconda3/envs/pytorch/bin/python download_data.py
## EfficientNet-B2 with RandAugment - 80.4 top-1, 95.1 top-5
## echo "I am startin training"
## /truba/home/efakhan/.conda/envs/timm/bin/python fake_train.py
# conda activate ingridient
# cd /truba/home/efakhan/foodstyles/ingredient-extraction
python run_ingredient_extraction.py --bert_model ./cased_L-12_H-768_A-12 --data_dir data/ --train_batch_size 24 --eval_batch_size 24 --num_train_epochs 10

