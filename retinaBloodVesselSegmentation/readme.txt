Retina blood vessel segmentation using PyTorch

Data preprocessing:
- load dataset: training, testing
- apply data augmentation on the training data (increase size of dataset)

Dataset
- 20x20 images
-test
	- 1st_manual
	- 2nd_manual
	- images
	- mask
-training
	- images: retina with vessels
	- 1st_manual: annotated mask
	- mask: circular mask, region of the retina (useful part)


- Implementation order
1. data
2. model
3. train
4. 