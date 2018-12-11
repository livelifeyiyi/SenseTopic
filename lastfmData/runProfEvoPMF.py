# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from profileEvolution_PMF import PMF

if __name__ == "__main__":
	rootDir = './'  # 'E:\\code\\SN2\\lastfm-2k\\'
	'''from topic_nan_id import nan_id  # 主题模型中取值nan的文档id号
	print("Reading Actual_rij_t.npy file......")
	Rij = np.load(rootDir + 'Rijt_rating_nozero.npy')  # Rij
	id_num = Rij.shape[0]
	data_t = []
	for idd in range(id_num):
		item_id = Rij[idd, 1]
		if item_id not in nan_id:
			data_t.append(Rij[idd])
	ratings = np.array(data_t)
	np.save("Rijt_rating_nozero_nonan.npy", ratings)
	'''
	ratings = np.load(rootDir+"Rijt_rating_nozero_nonan.npy")

	pmf = PMF(topic_file=rootDir+"lastfmDMM_topic5.theta", topic_type="LSTM", time_num=8, rootDir=rootDir)
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