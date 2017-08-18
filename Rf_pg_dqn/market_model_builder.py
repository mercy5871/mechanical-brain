from model_builder import AbstractModelBuilder
import numpy as np

class MarketPolicyGradientModelBuilder(AbstractModelBuilder):

	def buildModel(self):
		from keras.models import Model
		from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, concatenate
		from keras.layers.advanced_activations import LeakyReLU

		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		inputs = [B]
		merges = [b]

		for i in range(1):
			S = Input(shape=[2, 60, 1])
			inputs.append(S)

			h = Convolution2D(2048, (1, 3), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, (1, 5), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, (1, 10), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, (1, 20), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, (1, 40), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(512)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

			h = Convolution2D(2048, (1, 60), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(512)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

		m = concatenate(merges, axis = 1)
		m = Dense(1024)(m)
		m = LeakyReLU(0.001)(m)
		m = Dense(512)(m)
		m = LeakyReLU(0.001)(m)
		m = Dense(256)(m)
		m = LeakyReLU(0.001)(m)
		V = Dense(2, activation = 'softmax')(m)
		model = Model(inputs = inputs, outputs = V)

		return model

class MarketModelBuilder(AbstractModelBuilder):

	def buildModel(self):
		from keras.models import Model
		from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, concatenate
		from keras.layers.advanced_activations import LeakyReLU

		dr_rate = 0.0

		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		inputs = [B]
		merges = [b]

		for i in np.arange(1):
			S = Input(shape=[2, 60, 1])
			inputs.append(S)

			# h = Convolution2D(64, (1, 3), padding = 'valid')(S)
			# h = LeakyReLU(0.001)(h)
			# h = Convolution2D(128, (1, 5), padding = 'valid')(S)
			# h = LeakyReLU(0.001)(h)
			# h = Convolution2D(256, (1, 10), padding = 'valid')(S)
			# h = LeakyReLU(0.001)(h)
			h = Convolution2D(512, (1, 20), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			# h = Convolution2D(1024, (1, 40), padding = 'valid')(S)
			# h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(2048)(h)
			h = LeakyReLU(0.001)(h)
			h = Dropout(dr_rate)(h)
			merges.append(h)

			h = Convolution2D(2048, (1, 60), padding = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(4096)(h)
			h = LeakyReLU(0.001)(h)
			h = Dropout(dr_rate)(h)
			merges.append(h)

		m = concatenate(merges, axis = 1)
		m = Dense(1024)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		m = Dense(512)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		m = Dense(256)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		V = Dense(2, activation = 'linear', kernel_initializer = 'zero')(m)
		model = Model(inputs = inputs, outputs = V)

		return model
