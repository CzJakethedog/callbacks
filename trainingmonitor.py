# import the necessary packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import decimal
import os

# define the encode to complex
def encode_complex(obj):
     if isinstance(obj, complex):
        return [obj.real, obj.imag]

class TrainingMonitor(BaseLogger):
	def __init__(self, figPath, jsonPath=None, startAt=0):
		# store the output path for the figure, the path to the JSON
		# serialized file, and the starting epoch
		super(TrainingMonitor, self).__init__()
		self.figPath = figPath  # the path to the output plot that use to visualize loss and acc over time
		self.jsonPath = jsonPath # used to serialize the loss and acc_value as a JSON file 
		self.startAt = startAt # strating epch that training is resumed at when using "ctrl + c"

	def on_train_begin(self, logs={}):
		# initialize the history dictionary
		self.H = {}

		# if the JSON history path exists, load the training history
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				self.H = json.loads(open(self.jsonPath).read())

				# check to see if a strating epoch was supplied
				if self.startAt > 0:

					for k in self.H.keys():
						self.H[k]=self.H[k][:self.startAt]

	def on_epoch_end(self, epoch, logs={}):
		# loop over the logs and update the loss, accuracy, etc.
		# for the entire training process
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = l

		# Check to see the training history
		if self.jsonPath is not None:
			f = open(self.jsonPath, "w")
			f.write(json.dumps(self.H, default=encode_complex))
			f.close()

		# ensure at least two epochs have passed before plotting
		if len(self.H["loss"]) > 1:
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="train_loss")
			plt.plot(N, self.H["val_loss"], label="val_loss")
			plt.plot(N, self.H["acc"], label="train_acc")
			plt.plot(N, self.H["val_acc"], label="val_acc")
			plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()

			# save
			plt.savefig(self.figPath)
			plt.close()

