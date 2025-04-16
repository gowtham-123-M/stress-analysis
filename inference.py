"""
	Date : 07/10/2022

	AboutFile:
		> Gives the scoring the dataset
		> It will gives the predicted stages.
"""
import os
import pandas as pd
from model import Model
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Inference:

	def __init__(self):
		self.model_obj = Model()
		self.labels = None


	def predict(self):
		'''
		 	- get the features and labels.
		 	- send the processed the features to the dataset.
		'''
		pred_labels = []

		self.model_obj.get_threholds()
		self.model_obj.get_features_labels()
		features, self.labels = self.model_obj.features, self.model_obj.labels
		l = len(features)
		features_dict = features.T.to_dict()
		for i in tqdm(range(l)):
			entry = features_dict[i]
			pre_list , pre = self.model_obj.classification_rule(entry)
			pred_labels.append(pre)

		return pred_labels

	def get_score(self, y_true, y_pred):
		score  = accuracy_score(y_true, y_pred)
		return score


if __name__ == "__main__":
	infer_obj = Inference()

	y_pred = infer_obj.predict()
	y_true = infer_obj.labels

	score  = infer_obj.get_score(y_true, y_pred)
	print("\n Accuracy score : ",score)
	print("\n")
		


		
