# H2CGAN
We provide a simple GAN based baseline for image deraining, dehazing tasks. Some result images are uploaded in the image folder. Processing of cityscape images using H2CGAN and other SOA techniques are provided

A network trained with Cityscapes Images have been uploaded at https://drive.google.com/drive/folders/1CVAAVqmfXwqlTfQyALyVCwezwl19hjHc?usp=sharing 

The files should be downloaded in the parent git folder under checkpoints/dhcs4

To test cityscape images (https://www.cityscapes-dataset.com/file-handling/?packageID=31) from (https://www.cityscapes-dataset.com/downloads/) you need to download the images in the dataset/test1 folder.

To run the code:

mkdir dataset/test0

python3 test.py

Results will be saved in the results folder. Sample images are provided for testing. The code is tested in Python 3.8, torch 1.7.1, torchvision 0.8.2.







# Visualizing feature activations using the visualize notebook

To check feature distance in euclidean space among clean, hazy and dehazed images, save the .pt files in the "test" folder. Each image features are saved as a separate .pt file. Features can be extracted from Encoder of H2CGAN. Closer the distance between clean and hazy, better the encoder has learned the image features. Closer the distance between clean and dehazed, better the decoder has learnt to generate clean images from the hazy features.

a. To calculate Euclidean distance add the path of the .pt files (test folder)

b. It will generate distance values in .txt file as shown in the samples (oh,op)

c. Pass the .txt file in graph code to plot the graph.


To plot the feature activation sum plot RIS-original, RIS-DID and RIS-DRRID ,sample data is in test1 folder. Populate the folders with .pt files for each image. Features can be extracted from object detector.

a. Add the path of .pt files for all the three datasets.

b. Generate .csv files denotive sum of all the values>0.5 and values<0.5.

c. pass these .csv to generate the graph.


# Cite

-The paper "H2CGAN: Manageble AI for scene understanding tasks in hazy/rainy environment" is submitted to TIP IEEE. Authors are Pragya Mishra, Jhilik Bhattacharya, RK Sharma and Giovanni Ramponi 

-The code has been adapted from general cycleGANs and plexer architecture.
