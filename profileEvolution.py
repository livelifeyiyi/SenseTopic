import argparse
import codecs
import numpy as np
import ConnectDB
import random


class ProfileEvolution:
	def __init__(self, dbip, dbname, pwd, topic_file, learning_rate=0.01, minibatch=1000, max_iter=1000, feature_dimension=50, user_num=10000, time_num=30):
		self.D = feature_dimension
		conDB = ConnectDB.ConnectDB(dbip, dbname, pwd)
		self.cursor, self.db = conDB.connect_db()
		self.topic_file = topic_file  # M*1
		self.user_num = user_num
		self.time_num = time_num
		self.max_iter = max_iter
		self.minibatch = minibatch
		self.learning_rate = learning_rate

		# read topic assignment file
		topic_assign = []
		for doc_id, topic_num in enumerate(codecs.open(self.topic_file, mode='r', encoding='utf-8')):
			topic = [0 for i in range(self.D)]
			topic[topic_num] = 1
			topic_assign.append(topic)
		self.doc_num = len(topic_assign)   # M
		# topic_assign.shape = (self.doc_num, self.D)  # M*D
		self.topic_assign_np = np.array(topic_assign).T  # D*M
		# initialize user interest score: U
		self.user_interest = np.ones((self.time_num, self.D, self.user_num))

	def SGD_Uit(self, gamma, eta, lambda_U):
		for time in range(self.time_num):  # t in time_sequence
			for user in range(self.user_num):  # i in (N)
				Uit = self.minimum_Uit(user, time, gamma, eta, lambda_U)
				self.update_U_it(Uit, user, time)
		np.save('U_user_interest.npy', self.user_interest)
		# for j in self.doc_num(M)

	def minimum_Uit(self, user, time, gamma, eta, lambda_U):  # , item_set
		# user i(id),  time t
		user_h = self.neighbors(user, time, 0)
		gamma_i = gamma[user]

		sum_userh = 0.0
		for h in user_h:
			gamma_h = gamma[h]
			eta_h = eta[h]
			sum_userh += gamma_h * self.L_hit(h, user, time, eta_h) * (self.Uit_hat(h, time + 1) - self.U_it(h, time + 1))
		min_Uit = 0.0
		item_set = []
		for minb in range(self.minibatch):
			item_set.append(random.randint(0, self.doc_num))  # choose mini_batch number of documents' ids
		for iter in range(self.max_iter):
			for item in item_set:
				Y_ijt, R_ijt = self.Y_R_ijt(user, item, time)
				# item j
				min_Uit += Y_ijt * (np.dot(self.U_it(user, time), self.V_j(item)) - R_ijt) * self.V_j(item)
				+ lambda_U*(self.U_it(user, time)-self.Uit_hat(user, time, gamma_i))
				+ lambda_U*(1-gamma_i)(self.U_it(user, time+1)-self.Uit_hat(user, time+1, gamma_i))
				+ lambda_U*sum_userh

			min_Uit += self.learning_rate * min_Uit
			self.update_U_it(min_Uit, user, time)
			print min_Uit
		return min_Uit

	def minimum_gamma(self, user, lambda_U, time_sequence):
		# user i
		gamma_i = 0  # sample gamma to minimum target function

		sum_t = 0.0
		for t in time_sequence:
			neighbors_i = self.neighbors(user, t - 1, 0)
			sum_h = 0.0
			for h in neighbors_i:
				sum_h += self.L_hit(h, user, t - 1) * self.U_it(h, t - 1)
			sum_t += (self.Uit_hat(user, t, gamma_i) - self.U_it(user, t)) * (sum_h-self.U_it(user, t - 1))
		min_target = lambda_U * sum_t

	def minimum_eta(self, user_i, lambda_U, time_sequence, gamma):
		sum_t = 0.0
		gamma_i = gamma[user_i]
		for t in time_sequence:
			neighbors_i = self.neighbors(user_i, t - 1, 0)
			sum_h = 0.0
			for user_h in neighbors_i:

				is_friend = 0  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends
				friends_i = self.neighbors(user_i, t, 1)
				friends_h = self.neighbors(user_h, t, 1)
				intersec = list(set(friends_i).intersection(set(friends_h)))
				sum_h += (is_friend - float(len(intersec) / len(friends_i))) * self.U_it(user_h, t-1)

			sum_t += (self.Uit_hat(user_i, t, gamma_i) - self.U_it(user_i, t)) * (gamma_i * sum_h + (1-gamma_i) * self.U_it(user_, t-1))
		min_target = lambda_U * sum_t

	def Y_R_ijt(self, user_i, item_j, time):
		# Y_ijt = 1 if user_i has a link with item_j at time t, else=0
		# R_ijt = rating preference score of user_i to item_j at time t
		# `type` tb_miduserrelation_selected  # SELECT * FROM tb_miduserrelation
		sql = """SELECT `type` FROM tb_miduserrelation_selected
				WHERE `:START_ID`=%s AND `:END_ID`=%s AND `time`="%s" """ % (user_i, item_j, time)
		self.cursor.execute(sql)
		ress = self.cursor.fetchall()
		# print res
		Y_ijt = 0
		R_ijt = 0
		if len(ress) == 0:
			Y_ijt = 0
			R_ijt = 0
		else:
			for res in ress:
				Y_ijt = 1
				relation_type = res[0]
				if relation_type == 0:
					R_ijt = 1
				elif relation_type == 1:
					R_ijt = 2
				break
		return Y_ijt, R_ijt

	def V_j(self, item):
		"""
		:param item: The id of item
		:return: V_j = topic of item_j
		"""
		V_j = self.topic_assign_np[:, item]  # D*1
		return V_j

	def U_it(self, user, time):
		U_it = self.user_interest[time][:, user]  # D*1
		return U_it

	def update_U_it(self, U_it, user, time):
		self.user_interest[time][:, user] = U_it  # D*1

	def Uit_hat(self, user, time, gamma_i):  # user i
		neighbors_i = self.neighbors(user, time-1, 0)
		sum_h = 0.0
		for h in neighbors_i:
			sum_h += self.L_hit(h, user, time-1) * self.U_it(h, time-1)
		Uit_hat = (1-gamma_i) * self.U_it(user, time-1) + gamma_i * sum_h
		return Uit_hat

	def neighbors(self, user, time, flag):
		# flag = 0 return all neighbors, =1 return only friends.
		neighbors = []  # list of the users who have a link with user_i

		sql = """SELECT * FROM graph_1month_selected WHERE 
			(`:START_ID`=%s or `:END_ID`=%s) and `build_time` = '%s'""" % (user, user, time)
		self.cursor.execute(sql)
		results = self.cursor.fetchall()
		if flag == 0:
			for res in results:
				user1, user2 = res[0], res[1]
				if user1 == user and user2 not in neighbors:
					neighbors.append(user2)
				if user2 == user and user1 not in neighbors:
					neighbors.append(user1)
		else:
			follows = []
			followed = []
			for res in results:
				user1, user2 = res[0], res[1]
				if user1 == user:
					follows.append(user2)
				if user2 == user:
					followed.append(user1)
			friends = list(set(follows).intersection(set(followed)))
			neighbors = friends
		return neighbors  # save into a global parameter?

	def L_hit(self, user_h, user_i, time, eta):
		is_friend = self.get_friend_type(user_h, user_i, time)  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends

		friends_i = self.neighbors(user_i, time, 1)
		friends_h = self.neighbors(user_h, time, 1)
		intersec = list(set(friends_i).intersection(set(friends_h)))
		L_hit = eta * is_friend + (1-eta) * float(len(intersec)/len(friends_i))
		return L_hit

	def get_friend_type(self, user1, user2, time):
		sql = """SELECT * FROM graph_1month_selected 
			WHERE((`:START_ID`=%s AND `:END_ID`=%s ) or (`:START_ID`=%s AND `:END_ID`=%s)) and `build_time` = '%s'""" % (user1, user2, user2, user1, time)
		self.cursor.execute(sql)
		ress = self.cursor.fetchall()
		if len(ress) == 0:
			return 0
		elif len(ress) == 1:
			return 0.5
		elif len(ress) == 2:
			return 1
		else:
			return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-dbpwd", help="Password of database")
	parser.add_argument("-dbIP", help="IP address of database")
	parser.add_argument("-topicFile", help="Topic assignment file")
	args = parser.parse_args()
	pwd = args.dbpwd
	dbip = args.dbIP
	topic_file = args.topicFile

	Profile = ProfileEvolution(dbip=dbip, dbname='db_weibodata', pwd=pwd, topic_file=topic_file, feature_dimension=50)
	Profile.Y_R_ijt(1227898, 3361644068075147, '2011-11-02-11:18:14')
