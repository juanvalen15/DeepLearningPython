import os 
import numpy as np
import cv2 # read, resize images
from glob import glob # extract images and mask path 
from tqdm import tqdm # progress bar
import imageio # read gif mask
from albumentations import HorizontalFlip, VerticalFlip, Rotate # data augmentation library 

""" create a directory """
def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def load_data(path):
	train_x = glob(os.path.join(path, "training", "images", "*.tif"))
	train_y = glob(os.path.join(path, "training", "1st_manual", "*.gif"))
	
	test_x = glob(os.path.join(path, "test", "images", "*.tif"))
	test_y = glob(os.path.join(path, "test", "1st_manual", "*.gif"))
	
	return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
	size = (512, 512)

	for idx, (x,y) in tqdm(enumerate(zip(images, masks)),total=len(images)):
		""" extracting the name """
		name = x.split("/")[-1].split(".")[0]
		print(name)

		""" reading image and mask """
		x = cv2.imread(x, cv2.IMREAD_COLOR)
		y = imageio.mimread(y)[0]
		#print(x.shape, y.shape)
		
		if augment == True: # data augmentation is applied to the image
			aug = HorizontalFlip(p=1.0)
			augmented = aug(image=x, mask=y)
			x1 = augmented["image"]
			y1 = augmented["mask"]

			aug = VerticalFlip(p=1.0)
			augmented = aug(image=x, mask=y)
			x2 = augmented["image"]
			y2 = augmented["mask"]

			aug = Rotate(limit=45,p=1.0) # 45 deg
			augmented = aug(image=x, mask=y)
			x3 = augmented["image"]
			y3 = augmented["mask"]			
		
			X = [x, x1, x2, x3]
			Y = [y, y1, y2, y3]

		else: # no data augmentation is applied to the image
			X = [x] # list created to store images
			Y = [y] # list created to store masks

		index = 0
		for i, m in zip(X, Y):
			i = cv2.resize(i, size)
			m = cv2.resize(m, size)

			tmp_image_name = f"{name}_{index}.png"
			tmp_mask_name = f"{name}_{index}.png" ## JUAN: mask_name ?

			image_path = os.path.join(save_path, "image", tmp_image_name) # paths to store temp images
			mask_path = os.path.join(save_path, "mask", tmp_mask_name) # paths to store temp masks

			cv2.imwrite(image_path, i)
			cv2.imwrite(mask_path, m)

			index += 1
 

		# break # just to test on one image


if __name__ == "__main__":
	""" Seeding """
	np.random.seed(42)

	""" Load the data """
	data_path = "/Users/juan/Documents/GitHub/DeepLearningPython/retinaBloodVesselSegmentation/"
	(train_x, train_y), (test_x, test_y) = load_data(data_path)

	print(f"Train: {len(train_x)} - {len(train_y)}")
	print(f"Test: {len(test_x)} - {len(test_y)}")

	""" create directories to save the augmented data """
	create_dir("new_data/train/image/")
	create_dir("new_data/train/mask/") 
	create_dir("new_data/test/image/")
	create_dir("new_data/test/mask/") 

	""" Data augmentation """
	augment_data(train_x, train_y, "new_data/train/", augment=True)
	augment_data(test_x, test_y, "new_data/test/", augment=True)




