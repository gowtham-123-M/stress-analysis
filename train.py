"""
	Date : 11/10/2022

	AboutFile :
		> training the Machine learning Model
		> Here we are using the Decision trees 
		> Predicting stress and no stress.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from model import MLModel, DTModel
from load_data import Dataset
import pickle
from config import Hyperparameters as hp
import os
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.tree import plot_tree


class Train:

	def __init__(self):
		self.model_obj = MLModel()
		self.pred_mode = DTModel()
		self.data_obj = Dataset()
		self.x_train = None
		self.y_train = None
		self.x_val = None
		self.y_val = None
		self.model = None 
		self.output_results = "saved_models"


		self.cpp_alpha = None

	def load_model_data(self):
		self.model = self.model_obj.initialize_model()
		self.x_train, self.y_train = self.data_obj.getData(train=True)
		self.x_val, self.y_val = self.data_obj.getData(train=False)

	def savefigures(self,data, x_label = "x_label", y_label= "y_label", title = "title_name"):
		plt.figure(figsize=(20,15))
		plt.plot(data[0],data[1])
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.title(title)
		path = os.path.join(self.output_results, title+".png")
		plt.savefig(path)

	def post_pruning(self):
		
		print("|> Started Post Pruning Tree ....")

		clf = DecisionTreeClassifier()
		## asserting the message when there is noe train file and y_train files
		assert self.x_train is not None and self.y_train is not None, "first run the data loader method to get Training Dataset."
		assert self.x_val is not None and self.y_val is not None , "Run the dataloader for getting Validation Dataset."

		path = clf.cost_complexity_pruning_path(self.x_train, self.y_train)
		#ccp_alphas , impurities = path.ccp_alphas, path.impurities

		## saving the alpha vs impurities figure 
		"""self.savefigures([ccp_alphas, impurities], x_label="effective alpha", 
						y_label = "total impurities of leaves",
						title = "alpha_vs_imputies")"""

		clfs = []
		ccp_alphas = list(range(1,50,4))
		for ccp_alpha in tqdm(ccp_alphas):
			clf = KNeighborsClassifier(random_state=0, n_neighbors = ccp_alpha,algorithm='brute')
			clf.fit(self.x_train, self.y_train)
			clfs.append(clf)


		## saving the Effective Alphas and Total Depth
		#tree_depths = [clf.tree_.max_depth for clf in clfs]
		#self.savefigures([ccp_alphas[:-1], tree_depths[:-1]], "effective alphas", "total depth", "alphas_vs_depth")

		acc_scores = [accuracy_score(self.y_val, clf.predict(self.x_val)) for clf in clfs ]

		#tree_depths = [clf.tree_.max_depth for clf in clfs]

		## saving the alpha verses roc_auc_scores.
		self.savefigures([ccp_alphas[:-1], acc_scores[:-1]], "effective k value", "accuracy_score", "K_vs_Accuracy")

		scores = np.array(acc_scores[:-1])
		index = np.argmax(scores)

		self.ccp_alpha = ccp_alphas[index]











	def training(self):
		self.model.fit(self.x_train, self.y_train)

	def evaluation(self):
		y_train_pre = self.model.predict(self.x_train)
		y_val_pre = self.model.predict(self.x_val)

		## getting the training and validation scores.
		print("\n\n")
		print("> Accuracy Score : ")
		print("\t> training -- ", accuracy_score(self.y_train, y_train_pre))
		print("\t> validation -- ",accuracy_score(self.y_val, y_val_pre))

		print("\n> F1-Score : ")
		print("\t> training -- ", f1_score(self.y_train, y_train_pre))
		print("\t> validation -- ",f1_score(self.y_val, y_val_pre))
		print("\n\n")


	def evaluate_rule(self):
		x_tr = self.x_train.tolist()
		x_te = self.x_val.tolist()

		y_train_pre = []
		print("---predicting Train data----")
		for x in tqdm(x_tr):
			p = self.pred_mode.predict(x)
			y_train_pre.append(p)

		y_val_pre = []
		print("---- Predicting the Validation data -----")
		for x in tqdm(x_te):
			p = self.pred_mode.predict(x)
			y_val_pre.append(p)


		print("********** PREDICTION FROM RULE BASED MODEL **************")
		print("\n")
		print("> Accuracy Score : ")
		print("\t> training -- ", accuracy_score(self.y_train.tolist(), y_train_pre))
		print("\t> validation -- ",accuracy_score(self.y_val.tolist(), y_val_pre))

		print("\n> F1-Score : ")
		print("\t> training -- ", f1_score(self.y_train.tolist(), y_train_pre))
		print("\t> validation -- ",f1_score(self.y_val.tolist(), y_val_pre))
		print("\n\n")

		


	def saved_model(self):
		model_name = "dt_model.sav"
		out_path = os.path.join(hp.model_output_path,model_name)
		pickle.dump(self.model, open(out_path,"wb"))
		plt.figure(figsize = (40,40))
		plot_tree(self.model)
		path = os.path.join(self.output_results, "model_dt"+".png")
		plt.savefig(path)




if __name__ == "__main__":
	train_obj = Train()
	train_obj.load_model_data()
	# train_obj.post_pruning()
	# print("\n\t cpp_alpha : ",train_obj.ccp_alpha)
	train_obj.training()
	train_obj.evaluation()
	#train_obj.evaluate_rule()
	train_obj.saved_model()