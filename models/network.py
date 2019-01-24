
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from models.RoiPooling import RoiPoolingConv

	
class NET:
	def __init__(self):
		print ("network")

	def nn_base(self, input_tensor=None, trainable=False):


		input_shape = (None, None, 3)

		if input_tensor is None:
			img_input = Input(shape=input_shape)
		else:
			if not K.is_keras_tensor(input_tensor):
				img_input = Input(tensor=input_tensor, shape=input_shape)
			else:
				img_input = input_tensor

		bn_axis = 3

		# Block 1
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

		# Block 2
		x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
		x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		# Block 3
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
		x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		# Block 4
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

		# Block 5
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
		x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
		# x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

		return x

	def rpn_layer(self, base_layers, num_anchors):
		"""Create a rpn layer
			Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
					Keep the padding 'same' to preserve the feature map's size
			Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
					classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
					regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
		Args:
			base_layers: vgg in here
			num_anchors: 9 in here

		Returns:
			[x_class, x_regr, base_layers]
			x_class: classification for whether it's an object
			x_regr: bboxes regression
			base_layers: vgg in here
		"""
		x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

		x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
		x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

		return [x_class, x_regr, base_layers]



	def roi_pooling(self, pool_size, num_rois, x):
		assert(len(x) == 2)
		# import pdb; pdb.set_trace()
		img = x[0]
		rois = x[1]
		input_shape = K.shape(img)
		outputs = []
		nb_channels = img.shape[3]

		for roi_idx in range(num_rois):


			x = rois[0, roi_idx, 0]
			y = rois[0, roi_idx, 1]
			w = rois[0, roi_idx, 2]
			h = rois[0, roi_idx, 3]

			x = K.cast(x, 'int32')
			y = K.cast(y, 'int32')
			w = K.cast(w, 'int32')
			h = K.cast(h, 'int32')

			# Resized roi of the image to pooling size (7x7)
			rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (pool_size, pool_size))
			outputs.append(rs)
				

		final_output = K.concatenate(outputs, axis=0)

		# Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
		# Might be (1, 4, 7, 7, 3)
		final_output = K.reshape(final_output, (1, num_rois, pool_size, pool_size, nb_channels))

		# permute_dimensions is similar to transpose
		final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

		return final_output


	def classifier_layer_(self, base_layers, input_rois, num_rois, nb_classes=2):
		"""Create a classifier layer
		
		Args:
			base_layers: vgg
			input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
			num_rois: number of rois to be processed in one time (4 in here)

		Returns:
			list(out_class, out_regr)
			out_class: classifier layer output
			out_regr: regression layer output
		"""

		input_shape = (num_rois,7,7,512)

		pooling_regions = 7

		# out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
		# num_rois (4) 7x7 roi pooling
		out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
		# out_roi_pool = self.roi_pooling(pooling_regions, num_rois, [base_layers,input_rois])

		# Flatten the convlutional layer and connected to 2 FC and 2 dropout
		out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
		out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
		out = TimeDistributed(Dropout(0.5))(out)
		out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
		out = TimeDistributed(Dropout(0.5))(out)

		# There are two output layer
		# out_class: softmax acivation function for classify the class name of the object
		# out_regr: linear activation function for bboxes coordinates regression
		out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
		# note: no regression target for bg class
		out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

		return [out_class, out_regr] 



	def classifier(self, base_layers, input_rois, num_rois, nb_classes = 2, trainable=False):

	    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

	    if K.backend() == 'tensorflow':
	        pooling_regions = 7
	        input_shape = (num_rois,7,7,512)
	    elif K.backend() == 'theano':
	        pooling_regions = 7
	        input_shape = (num_rois,512,7,7)
	    import pdb; pdb.set_trace()
	    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

	    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
	    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
	    out = TimeDistributed(Dropout(0.5))(out)
	    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
	    out = TimeDistributed(Dropout(0.5))(out)

	    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
	    # note: no regression target for bg class
	    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

	    return [out_class, out_regr]