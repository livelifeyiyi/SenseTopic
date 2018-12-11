import argparse
import json
import codecs
import numpy as np
import math
import heapq


class evaluatePrediction:
	def __init__(self, rootDir, time_num):
		self.rootDir = rootDir
		# self.user_num = int(user_num)
		# self.item_mid_map = np.loadtxt(mid_dir)
		self.time_num = time_num
		print("Reading Actual_rij_t.npy file......")
		self.R_ij = np.load(self.rootDir + 'Actual_Rij_t.npy')
		print("Reading Predicted Rij file......")
		self.pred_R_ij = np.load(self.rootDir + 'Predict_Rij_t' + str(time_num)+'.npy')  #  + '/predict_lambda1/Predict_Rij_t'
		# with codecs.open(self.rootDir + '/Predict_Rij_t' + str(time_num)+'.json', mode='r') as infile:
		# 	self.pred_R_ij = json.load(infile)
			# self.pred_R_ij = json.load(self.rootDir + '/Predict_Rij_t' + str(time_num)+'.json')
		# for test
		# self.R_ij = np.ones((self.time_num, self.user_num, self.doc_num), dtype='int')
		# self.R_ij = np.load(self.rootDir + 'Actual_Rij_1.npy')
		# self.pred_R_ij = np.load(self.rootDir + 'predict/Predict_Rij_t' + str(time_num) + '.npy')

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
				pred_rij = self.pred_R_ij[user_id][item_id] / 10000.0  #
				if np.isnan(pred_rij):
					pred_rij = 0.0
				act_rij = self.R_ij[time][user_id][item_id]
				# act_rij = self.R_ij[user_id][item_id]
				tmp = (pred_rij-act_rij) ** 2
				item_sum += tmp
				tmp_mae = pred_rij-act_rij
				# tmp_mae = math.fabs(pred_rij-act_rij)
				item_sum_mae += tmp_mae
			print(math.sqrt(item_sum/item_num), item_sum_mae/item_num)
			user_sum += item_sum
			user_sum_mae += item_sum_mae

		print("ALL RMSE.......")
		print(math.sqrt(user_sum/(user_num*item_num)), user_sum_mae/(user_num*item_num))

	def MRR(self):
		time = self.time_num
		print("Time: " + str(time))
		user_num = len(self.pred_R_ij)
		'''user_sum = 0.0
		user_sum_mae = 0.0
		item_num = len(self.pred_R_ij[0])'''
		totalsum = []
		totalsum_accu = []
		for user_id in range(user_num):  # i in (N)
			# print("user_id: " + str(user_id))
			'''item_sum = 0.0
			item_sum_mae = 0.0'''
			act_ri = self.R_ij[int(time)][user_id]
			# act_ri = self.R_ij[user_id]
			if sum(act_ri) == 0:
				continue
			else:
				act_rij_nozero = {}
				pred_ri = self.pred_R_ij[user_id]
				num = 0.0
				sumv = 0.0
				accurate = 0.0
				pred_ri_sorted = sorted(pred_ri, reverse=True)

				for item_id, R_ij in enumerate(act_ri):
					if R_ij != 0:
						num += 1
						act_rij_nozero[item_id] = R_ij
						'''pred_ij = pred_ri[item_id]
						if np.isnan(pred_ij):
							pass
							# print item_id, R_ij, pred_ij
						else:
							mrr_loc = pred_ri_sorted.index(pred_ij) + 1
							sumv += 1.0/mrr_loc
							# print item_id, R_ij, pred_ij, mrr_loc'''
				maxn = pred_ri_sorted[0:30]  # 0
				item_num = 0
				for index, valuen in enumerate(maxn):
					maxn_itemid = [i for i, j in enumerate(pred_ri) if j == valuen]

					for pred_itemid in maxn_itemid:
						item_num += 1
						if pred_itemid in act_rij_nozero.keys():
							sumv += 1.0/(index+1)
							accurate += 1.0
				if item_num != 0 and num != 0 and sumv != 0:
					print accurate/item_num, 1.0/num*sumv
					totalsum.append(1.0/item_num)  # num*sumv
					totalsum_accu.append(accurate/item_num)
				'''
				pred_ri_max_index = map(list(pred_ri).index, heapq.nlargest(len(act_rij_nozero), pred_ri))
				accurate = 0.0
				for i in pred_ri_max_index:
					if i in act_rij_nozero.keys():
						accurate += 1
				print accurate/len(act_rij_nozero)'''
		print("Accuracy: " + str(np.mean(np.array(totalsum_accu))))
		print("MRR:" + str(np.mean(np.array(totalsum))))

if __name__ == '__main__':
	# rootdir = 'E:/code/SN2/lastfm-2k/'
	parser = argparse.ArgumentParser()
	parser.add_argument("-time", help="Time number")
	args = parser.parse_args()

	time_id = args.time
	# evaluate = evaluatePrediction('E:\\code\\SN2\\lastfm-2k\\', "E:\\code\\SN2\\lastfm-2k\\artistsID_id_havetags_havevecs.txt", 1)
	# for time_id in range(0, 7):
	evaluate = evaluatePrediction('./', time_id)  #  rootdir+"./artistsID_id_havetags_havevecs.txt"
	# evaluate.RMSE()
	evaluate.MRR()
	# evaluate.evaluate_test()