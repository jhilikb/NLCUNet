# NLCUNET
We provide a simple GAN based baseline with nonlinear block

The dataset list can be found here:-
| Task | Name | Link |
|----------|----------|----------|
| Dehazing    | SOTS   | [SOTS](https://sites.google.com/view/reside-dehaze-datasets/reside-v0?authuser=0)   |
| Dehazing    | Cityscapes  |[City](https://www.cityscapes-dataset.com/file-handling/?packageID=31)   |
| Night2Day    | Robocar   | [Aachen Day-Night datasets ](https://www.visuallocalization.net/datasets/)  |
|Low2Bright    | LOL   | [LOL](https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB)   |



The pretrained models can be found here:
| Task | Trained=on | Name |
|----------|----------|----------|
| Dehazing    | OTS   | [sotv2c](https://drive.google.com/drive/folders/1CVAAVqmfXwqlTfQyALyVCwezwl19hjHc?usp=sharing)   |
| Dehazing    | SUN  |[csv2c](https://drive.google.com/drive/folders/1CVAAVqmfXwqlTfQyALyVCwezwl19hjHc?usp=sharing)   |
| Night2Day    | Robocar   | [n2dv2c ](https://drive.google.com/drive/folders/1CVAAVqmfXwqlTfQyALyVCwezwl19hjHc?usp=sharing)  |
|Low2Bright    | Adobe   | [l2bv2c](https://drive.google.com/drive/folders/1CVAAVqmfXwqlTfQyALyVCwezwl19hjHc?usp=sharing)   | https://drive.google.com/drive/folders/1CVAAVqmfXwqlTfQyALyVCwezwl19hjHc?usp=sharing 

The files should be downloaded in the parent git folder under checkpoints/


To run the code:

mkdir dataset/test0

Test your code using the task name and the model name. For example if you want to use the pretrained model trained with SUN dataset for dehazing use:
python3 test.sh --task dehazing --model sun

The different task names are dehazing,deraining,low2bright and night2day.

The different model names are sun, ots , adobe , robocar representing the dataset it is trained on.

Results will be saved in the results folder. Sample images are provided for testing. The code is tested in Python 3.8, torch 1.7.1, torchvision 0.8.2.






# Cite

Please cite the paper as: 
J. Bhattacharya, A. Carini, S. Marsi and G. Ramponi, "A Polynomial and Fourier Basis Network for Vision-based Translation Tasks"

-The code has been adapted from general cycleGANs and plexer architecture.
