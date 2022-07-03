from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
	x = Conv2D(num_filters, 3, padding="same")(inputs)
	x = BatchNormalization()(x)