import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # to reduce keras messages
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_loss, dice_coef, iou

H = 512
W = 512

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def load_data(path):
	x = sorted(glob(os.path.join(path, "image", "*.png"))) # is this the right format?
	y = sorted(glob(os.path.join(path, "mask", "*.png"))) # is this the right format?
	return x, y


def shuffling(x, y):
	x, y = shuffle(x, y, random_state=42)
	return x, y

def read_image(path):
	path = path.decode()
	x = cv2.imread(path, cv2.IMREAD_COLOR)
	#x = cv2.resize(x, (W, H))
	x = x/255.0
	x = x.astype(np.float32)
	return x

def read_mask(path):
	path = path.decode()
	x = cv2.imread(path, cv2.IMREAD_GRAYSCALE) ## (512, 512)
	#x = cv2.resize(x, (W, H))
	x = x/255.0
	x = x.astype(np.float32)
	x = np.expand_dims(x, axis=-1) ## (512, 512, 1)
	return x

def tf_parse(x, y):
	def _parse(x, y):
		x = read_image(x)
		y = read_mask(y)
		return x, y

	x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
	x.set_shape([H, W, 3])
	y.set_shape([H, W, 1])
	return x, y

def tf_datase(X, Y, batch_size=2):
	dataset = tf.data.Dataset.from_tensor_slices((X, Y))
	dataset = dataset.map(tf_parse)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(4)
	return dataset


# ------ Main ------ #
if __name__ == "__main__":
	""" Seeding """
	np.random.seed(42)
	tf.random.set_seed(42)

	""" Directory to save files """
	create_dir("/Users/juan/Documents/GitHub/DeepLearningPython/retinaBloodVesselSegmentation/files")

	""" Hyperparameters """
	batch_size = 2
	lr = 1e-4
	num_epochs = 100
	model_path = os.path.join("files", "model.h5")
	csv_path = os.path.join("files", "data.csv")

	""" Dataset """
	dataset_path = "/Users/juan/Documents/GitHub/DeepLearningPython/retinaBloodVesselSegmentation/new_data"
	train_path = os.path.join(dataset_path, "train")
	valid_path = os.path.join(dataset_path, "test")

	train_x, train_y = load_data(train_path)	
	train_x, train_y = shuffling(train_x, train_y)
	valid_x, valid_y = load_data(valid_path)

	print(f"Train: {len(train_x)} - {len(train_y)}")
	print(f"Valid: {len(valid_x)} - {len(valid_y)}")











