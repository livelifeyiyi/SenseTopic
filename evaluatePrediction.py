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

	def evaluate_test(self):
		with codecs.open(self.rootDir + "Predict_Rij_t21.json", mode='r') as infile:
			Predict_Rij = json.load(infile)
			user_num = len(Predict_Rij["20"])
			for user_id in range(user_num):
				print("********user_id********" + str(user_id))
				Rij = Predict_Rij["20"]["%s" % user_id]
				for each_id in range(len(Rij)):
					if Rij[each_id] != 0.0:
						print("index: " + str(each_id) + ", Rij: " + str(Rij[each_id]))

	def RMSE(self):
		for time in range(20, self.time_num):  # t in time_sequence; count from 1
			print("Time: " + str(time))
			with codecs.open(self.rootDir + "Predict_Rij_t" + str(time) + ".json", mode='r') as infile:
				Predict_Rij = json.load(infile)
				user_num = len(Predict_Rij["20"])
				user_sum = 0.0
				user_sum_mae = 0.0
				for user_id in range(user_num):  # i in (N)
					item_num = len(Predict_Rij["20"][user_id])
					item_sum = 0.0
					item_sum_mae = 0.0
					for item_id in range(item_num):
						pred_rij = Predict_Rij["20"][user_id][item_id]
						act_rij = self.R_ijt(user_id, item_id, time)
						tmp = (pred_rij-act_rij) ** 2
						item_sum += tmp
						tmp_mae = math.fabs(pred_rij-act_rij)
						item_sum_mae += tmp_mae
					print(math.sqrt(item_sum/item_num), item_sum_mae/item_num)
					user_sum += item_sum
					user_sum_mae += item_sum_mae
				print("ALL RMSE.......")
				print(math.sqrt(user_sum/(user_num*item_num)), user_sum_mae/(user_num*item_num))

	def R_ijt(self, user_i, item_j, time):
		# R_ijt = rating preference score of user_i to item_j at time t
		# `type` tb_miduserrelation_selected  # SELECT * FROM tb_miduserrelation
		mid = self.item_mid_map[item_j]
		sql = """SELECT `type` FROM tb_miduserrelation_selected_time
				WHERE `:START_ID`=%s AND `:END_ID`=%s AND `time_index`="%s" """ % (user_i, mid, time)
		self.cursor.execute(sql)
		ress = self.cursor.fetchall()
		# print res
		R_ijt = 0
		if len(ress) == 0:
			R_ijt = 0
		else:
			for res in ress:
				relation_type = res[0]
				if relation_type == 0:
					R_ijt = 1
				elif relation_type == 1:
					R_ijt = 2
				break
		return R_ijt

if __name__ == '__main__':
	evaluate = evaluatePrediction('E:\\code\\SN2\\profile\\', "../data/mid_id_user100", 21)
	evaluate.RMSE()