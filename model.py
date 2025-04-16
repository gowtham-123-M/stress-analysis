"""
	Date : 07/10/2022

	AboutFile:
		> Building the Rule using Thresholds.
		> Thresholds are choose from the research paper
"""
import os
from load_data import Dataset
import numpy as np
from config import Hyperparameters as hp
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)

class Model:

	def __init__(self):
		self.data_obj = Dataset()
		self.labels = None
		self.features = None
		self.thresholds = None
		self.swell_thresholds = None

	def get_features_labels(self):
		df = self.data_obj.load_data()
		self.features = df.iloc[:,:-1]
		self.labels = df.iloc[:,-1].values
		self.labels = (self.labels>=1).astype(int).tolist()

	def get_threholds(self):
		thre = {}
		thre["MEAN_RR"] = 697.15
		thre["SDRR"] = 46.92
		thre["RMSSD"] = 35.92
		thre["SDSD"] = 35.92
		thre["pNN25"] = 41.26
		thre["pNN50"] = 12.3
		thre["LF"] = 823.03
		thre["HF"] = 589.56
		thre["LF_HF"] = 1.82

		self.thresholds = thre

		thre = {}
		thre["MEAN_RR"] = 849.7265535
		thre["SDRR"] = 109.844804
		thre["RMSSD"] = 15.050675
		thre["SDSD"] = 15.0499495
		thre["pNN25"] = 9.993337499999999
		thre["pNN50"] = 0.877282
		thre["LF"] = 953.3702089999999
		thre["HF"] = 38.8555955
		thre["LF_HF"] = 120.9352995

		self.swell_thresholds = thre

#["MEAN_RR", "SDRR", "RMSSD", "SDSD", "pNN25", "pNN50", "LF", "HF", "LF_HF", "condition"]

	def classification_rule(self, dt = None):
		'''
			Here we are defining the 2 stage.
				--> Medium stress -- 0
				--> High stress -- 1
		'''

		if dt is None:
			print("\n > There was no input for the ")
			exit(0)

		stress_level = []

		if(not hp.USE_SWELL_MODEL):
			## ---- Level for MEAN_RR
			if(dt["MEAN_RR"] < self.thresholds["MEAN_RR"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)

			## ----- level for SDRR ------
			if(dt["SDRR"] > self.thresholds["SDRR"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ----- level for RMSSD -------
			if(dt["RMSSD"] > self.thresholds["RMSSD"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## --------- level for SDSD ---------
			if(dt["SDSD"] > self.thresholds["SDSD"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## -------- level for pNN25 -------
			if(dt["pNN25"] < self.thresholds["pNN25"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ------- level for pNN50 --------
			if(dt["pNN50"] > self.thresholds["pNN50"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ------- level for LF -------
			if(dt["LF"] > self.thresholds["LF"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ------ level for the HF
			if(dt["HF"] > self.thresholds["HF"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ---- level for the ratio of LF & HF
			if(dt["LF_HF"] > self.thresholds["LF_HF"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


		else:
			## ---- Level for MEAN_RR
			if(dt["MEAN_RR"] > self.swell_thresholds["MEAN_RR"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)

			## ----- level for SDRR ------
			if(dt["SDRR"] > self.swell_thresholds["SDRR"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ----- level for RMSSD -------
			if(dt["RMSSD"] > self.swell_thresholds["RMSSD"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## --------- level for SDSD ---------
			if(dt["SDSD"] > self.swell_thresholds["SDSD"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## -------- level for pNN25 -------
			if(dt["pNN25"] > self.swell_thresholds["pNN25"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ------- level for pNN50 --------
			if(dt["pNN50"] > self.swell_thresholds["pNN50"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ------- level for LF -------
			if(dt["LF"] > self.swell_thresholds["LF"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ------ level for the HF
			if(dt["HF"] < self.swell_thresholds["HF"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


			## ---- level for the ratio of LF & HF
			if(dt["LF_HF"] > self.swell_thresholds["LF_HF"]):
				# high
				stress_level.append(1)
			else:
				stress_level.append(0)


		val = sum(stress_level)/len(stress_level) ## checking the value
		if(val> 0.5):
			return stress_level, 1
		else:
			return stress_level, 0




class MLModel:

	def __init__(self):
		## Initializing the parameters
		self.min_samples_split = 50#hp.min_samples_split
		self.ccp_alpha = 0.001
		self.max_depth=10

	def initialize_model(self):
		model = DecisionTreeClassifier(min_samples_split= self.min_samples_split, ccp_alpha= self.ccp_alpha, max_depth=self.max_depth)
		return model




class DTModel:

	def __init__(self):
		pass

	def predict(self, x):

		## root
		if(x[0] <= 819.596):

			## left subtreee 1
			if(x[0] <= 579.494):
				if(x[8]<= 10.176):
					class_ = 0
				else:
					class_ = 1

			else:
				# if(x[1] <= 103.628):
				# 	class_ = 0

				# else:
				# 	if(x[4] <= 16.767):
				# 		class_ = 0
				# 	else:
				# 		class_ =1

				class_ = 0


		else:

			## right subtree 1
			if(x[4] > 22.033):
				class_ = 0

			else:
				# left subtree 2
				if(x[0] <= 936.129):
					if(x[7] <= 17.311):
						## left subtree 3
						if(x[1] <= 72.454):
							class_ = 0
						else:
							if(x[3]<= 10.951):
								if(x[8] <= 30.651):
									class_ = 0

								else:
									class_ = 1

							else:
								class_ = 0

					else:
						## right subtree 3
						if(x[5] <= 0.433):
							if(x[8] <= 21.308):
								class_ = 1

							else:
								if(x[6] <= 1385.16):
									if(x[4] <= 6.233):
										if(x[5] <= 0.033):
											class_ = 0
										else:
											class_ = 1
									else:
										class_ = 0

								else:
									class_ = 1

						else:
							## right subtree 4
							if(x[1] <= 98.153):
								class_ = 0

							else:
								if(x[1] <= 247.592):
									if(x[5] <= 2.433):
										if(x[1] <= 118.713):
											class_ = 1
										else:
											class_ = 0

									else:
										class_ = 1

								else:
									class_ = 0

				else:
					### right subtree 2
					if(x[1] <= 284.215):
						if(x[6] <= 996.667):
							if(x[1] <= 92.88):
								if(x[2] <= 11.954):
									class_ = 1

								else:
									if(x[8] <= 362.722):
										class_ = 0

									else:
										class_ = 1

							else:
								if(x[0] <= 954.675):
									if(x[4] <= 9.167):
										class_ = 1

									else:
										class_ = 0

								else:
									class_ = 1

						else:
							if(x[8] <= 162.178):
								class_ = 1

							else:
								if(x[8] <= 2329.273):
									class_ = 0

								else:
									class_ = 1

					else:
						if(x[5] <= 1.833):
							class_ = 0

						else:
							class_ = 0

		return class_
















