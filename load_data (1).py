'''
	Date : 07/10/2022

	AboutFile:
		> loading data 
		> preprocessing the data
'''
import os
import glob
import sys
from config import Hyperparameters as hp
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Dataset(object):
	"""
		Loading the dataset for the training stress analysis.

	"""
	def __init__(self):
		super(Dataset, self).__init__()
		self.data_path = hp.PRO_DATA_PATH ## preprocessed dataset --> data contains in a single DataFrame,.
										  ## contains --> linear-analysis parameters & non-linear analysis parameters & labels. 
		self.raw_data_path = hp.RAW_DATA_PATH ## contains a rr-intervals & respective labels

		self.l2i = {'no stress':0, 'interruption':0, 'time pressure':1}
		self.i2l = {0:'no stress', 1:'interruption', 2:'time pressure'}

		self.linear_analysis_parms = ["MEAN_RR", "SDRR", "RMSSD", "SDSD", "pNN25", "pNN50", "LF", "HF", "LF_HF", "condition"]
		#self.linear_analysis_parms = ["MEAN_RR", "SDRR", "pNN50", "LF_HF", "condition"]


	def load_data(self):
		if(hp.WORKING == "preprocessed"):
			train_data_path = os.path.join(self.data_path, "train.csv")
			train_data = pd.read_csv(train_data_path)
			df = train_data[self.linear_analysis_parms]
			df["condition"] = df["condition"].map(self.l2i)
			return df

	## Here it will load both train and validation Dataset.
	def getData(self, train = True):
		if(train):
			path = os.path.join(self.data_path, "train.csv")
		else:
			path = os.path.join(self.data_path, "test.csv")


		data = pd.read_csv(path)
		df = data[self.linear_analysis_parms]
		df["condition"] = df["condition"].map(self.l2i)

		x =df.iloc[:,:-1].values
		y = df.iloc[:,-1].values

		return x,y

	



if __name__ == "__main__":
	data_obj = Dataset()
	df = data_obj.load_data()
	print(df.head())








