# NLCSine
We provide a simple GAN based baseline with nonlinear block

A network trained with Robocar Images have been uploaded at https://drive.google.com/drive/folders/1CVAAVqmfXwqlTfQyALyVCwezwl19hjHc?usp=sharing 

The files should be downloaded in the parent git folder under checkpoints/n2d

To test robocar images (https://www.cityscapes-dataset.com/file-handling/?packageID=31) from (https://www.cityscapes-dataset.com/downloads/) you need to download the images in the dataset/test1 folder.

To run the code:

mkdir dataset/test0

python3 test.py

Results will be saved in the results folder. Sample images are provided for testing. The code is tested in Python 3.8, torch 1.7.1, torchvision 0.8.2.






# Cite

Please cite the paper as: 
J. Bhattacharya, A. Carini, S. Marsi and G. Ramponi, "A Nonlinear Convolution Network for Image Enhancement and Translation Tasks"

-The code has been adapted from general cycleGANs and plexer architecture.
