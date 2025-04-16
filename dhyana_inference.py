"""
	Date : 11/10/2022

	AboutFile:
		> Input data : Heart rate at 1hz 
		> Convert the Heart rate to rr-intervals 
		> build the features for the rr-intervals
		> pass the features to the Model. 
"""
import pandas as pd
import numpy as np
import features
import os
import pickle
from config import Hyperparameters
from model import DTModel
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt




def moving_average(x):
	window_length = 7 ## window length need to be in odd.
	mode = "nearest"
	# x = x.tolist()
	l = len(x)

	
	## padding
	if(mode == "nearest"):
		padded_x = [x[0]]*(int(window_length//2)) + x + [x[-1]]*(int(window_length // 2))

	## Calculating the Moving Average ..
	xx = []
	for i in range(l):
		st = i
		ed = i+window_length
		m = sum(padded_x[st:ed])
		xx.append(m/window_length)
	return np.array(xx)

class FeatureBuilder:

	def __init__(self):
		self.data_path = os.path.join("test_data/harsha2_hr.csv")
		self.linear_analysis_parms = ["MEAN_RR", "SDRR", "RMSSD", "SDSD", "pNN25", "pNN50", "LF", "HF", "LF_HF"]


	def load_data(self):
		df = pd.read_csv(self.data_path)
		hr = df["HEART_RATE"]
		hr = moving_average(hr.tolist())
		rri = self.convert_rri(hr)
		return rri


	def convert_rri(self, hr):
		rri = 60000/hr
		return rri


	def build(self, rr_intervals):
		td_features = features.get_time_domain_features(rr_intervals)
		fd_features = features.get_frequency_domian_features(rr_intervals)
		return td_features, fd_features


	def parse_intervals(self, parse_interval = 60):
		rri = self.load_data()
		assert len(rri)>60, "Data is not enough to predict the stress"

		iters = len(rri)//parse_interval

		dfs = []
		for i in range(iters):
			df = {}
			st = i
			ed = i*st + parse_interval
			rr_intervals = rri[st:ed]

			tdf, fdf = self.build(rr_intervals)

			df["MEAN_RR"] = tdf["MEAN_RR"]
			df["SDRR"] = tdf["SDRR"]
			df["RMSSD"] = tdf["RMSSD"]
			df["SDSD"] = tdf["SDSD"]
			df["pNN25"] = tdf["pNN25"]
			df["pNN50"] = tdf["pNN50"]

			df["LF"] = fdf["LF"]
			df["HF"] = fdf["HF"]
			df["LF_HF"] = fdf["LF_HF"]
			# print(df)
			
			dfs.append(df)

		# print(dfs)
		df_ = pd.DataFrame(dfs)


		return df_



class Inference:

	def __init__(self):
		self.model_path = os.path.join("saved_models", "dt_model.sav")
		self.model = pickle.load(open(self.model_path,"rb"))
		self.pred_model= DTModel()

	def predict(self, x):
		y_test_pre = self.model.predict(x)
		return y_test_pre

	def predict_rule(self, x):
		xx = x.values.tolist()
		y_test_pre = []
		for i in xx:
			p = self.pred_model.predict(i)
			y_test_pre.append(p)
		return y_test_pre
    
   ## def graphs(self):
       ## x.hypnogram(y_test_pre)
      ##  x.hypnogram(y_test_pre1)
        
if __name__ == "__main__":
	fb_obj = FeatureBuilder()
	test_data = fb_obj.parse_intervals()

	test_data.fillna(0, inplace=True)
	test_data.to_csv("feat.csv",index=False)
	infer_obj = Inference()
	y_test_pre = infer_obj.predict(test_data)

	y_test_pre1 = infer_obj.predict_rule(test_data)
    
 
    

	print("\n")
	print(y_test_pre)

	print("\n")
	print(y_test_pre1)
	print("\n")
    
    
  
    
    
    
    