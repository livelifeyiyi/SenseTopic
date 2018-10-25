import json
import codecs
import numpy as np
import math


class evaluatePrediction:
	def __init__(self, rootDir, mid_dir, time_num):
		self.rootDir = rootDir
		# self.user_num = int(user_num)
		self.item_mid_map = np.loadtxt(mid_dir)
		self.time_num = time_num
		print("Reading Actual_rij_t.npy file......")
		# self.R_ij = np.ones((self.time_num, self.user_num, self.doc_num), dtype='int')
		# self.R_ij = np.load(self.rootDir + 'Actual_Rij_t.npy')
		print("Reading Predicted Rij file......")
		# self.R_ij = np.ones((self.time_num, self.user_num, self.doc_num), dtype='int')
		# self.pred_R_ij = np.load(self.rootDir + 'temp/Predict_Rij_t'+str(time_num)+'.npy')

	def evaluate_test(self):
		R_ij_1 = np.load(self.rootDir + 'Predict_Rij_t1.npy')
		# with codecs.open(self.rootDir + "Predict_Rij_t21.json", mode='r') as infile:
			# Predict_Rij = json.load(infile)
		user_num = len(R_ij_1)
		for user_id in range(user_num):
			Rij = R_ij_1[user_id]  # [time]

			# Rij = Predict_Rij["20"]["%s" % user_id]
			for each_id in range(len(Rij)):
				if Rij[each_id] != 0:
					print("********user_id********" + str(user_id))
					print("index: " + str(each_id) + ", Rij: " + str(Rij[each_id]))

	def RMSE(self):
		# for time in range(1, self.time_num+1):  # t in time_sequence; count from 1
		time = self.time_num
		print("Time: " + str(time))
		user_num = len(self.pred_R_ij)
		user_sum = 0.0
		user_sum_mae = 0.0
		item_num = len(self.pred_R_ij[0])
		for user_id in range(user_num):  # i in (N)
			item_sum = 0.0
			item_sum_mae = 0.0
			for item_id in range(item_num):
				pred_rij = self.pred_R_ij[user_id][item_id]  #
				act_rij = self.R_ij[time][user_id][item_id]
				tmp = (pred_rij-act_rij) ** 2
				item_sum += tmp
				tmp_mae = math.fabs(pred_rij-act_rij)
				item_sum_mae += tmp_mae
			print(math.sqrt(item_sum/item_num), item_sum_mae/item_num)
			user_sum += item_sum
			user_sum_mae += item_sum_mae
		print("ALL RMSE.......")
		print(math.sqrt(user_sum/(user_num*item_num)), user_sum_mae/(user_num*item_num))


if __name__ == '__main__':
	evaluate = evaluatePrediction('E:\\code\\SN2\\lastfm-2k\\', "E:\\code\\SN2\\lastfm-2k\\artistsID_id_havetags_havevecs.txt", 1)
	# evaluate = evaluatePrediction('./', "./artistsID_id_havetags_havevecs.txt", 1)

	# evaluate.RMSE()
	evaluate.evaluate_test()