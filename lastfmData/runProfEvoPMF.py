# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from profileEvolution_PMF import PMF

if __name__ == "__main__":
	rootDir = 'E:\\code\\SN2\\lastfm-2k\\'

	print("Reading Actual_rij_t.npy file......")
	'''Rij = np.load(rootDir + 'Actual_Rij_t.npy')  # Rij
	time_num, user_num, item_num = Rij.shape[0], Rij.shape[1], Rij.shape[2]
	data_t = []
	for time_id in range(time_num):
		for user_id in range(user_num):
			for item_id in range(item_num):
				rijt = Rij[time_id][user_id][item_id]
				if rijt != 0:
					data_t.append([user_id, item_id, rijt, time_id])
	ratings = np.array(data_t)
	np.save("Rijt_rating_nozero.npy", ratings)
	'''
	ratings = np.load(rootDir+"Rijt_rating_nozero.npy")

	pmf = PMF(topic_file=rootDir+"lastfmDMM.theta", topic_type="LSTM", time_num=8, rootDir=rootDir)
	# pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
	# 				"batch_size": 1000, "topic_file": "lastfmDMM.theta", "topic_type": "LSTM", "time_num": 8, "rootDir": rootDir})

	print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
	train, test = train_test_split(ratings, test_size=0.2)  # spilt_rating_dat(ratings)
	pmf.fit(train, test)

	# Check performance by plotting train and test errors
	plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
	plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
	plt.title('The MovieLens Dataset Learning Curve')
	plt.xlabel('Number of Epochs')
	plt.ylabel('RMSE')
	plt.legend()
	plt.grid()
	plt.show()
	print("precision_acc,recall_acc:" + str(pmf.topK(test)))