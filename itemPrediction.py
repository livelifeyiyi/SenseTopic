import argparse

import numpy as np
import ConnectDB
import codecs
import json
from selected_user import selected_user
selected_user = selected_user[0:100]
max_uid = selected_user[-1]

class itemPrediction:
	def __init__(self, dbip, dbname, pwd, topic_file, mid_dir, feature_dimension, user_num, time_num, iteround, rootDir):
		self.D = int(feature_dimension)
		conDB = ConnectDB.ConnectDB(dbip, dbname, pwd)
		self.cursor, self.db = conDB.connect_db()
		self.topic_file = topic_file  # M*1
		self.user_num = int(user_num)
		self.time_num = int(time_num)
		self.iteround = int(iteround)
		self.mid_dir = mid_dir
		self.rootDir = rootDir
		self.item_mid_map = np.loadtxt(self.mid_dir)
		print("Reading the topic assignment file......")
		topic_assign = []
		for doc_id, topic_num in enumerate(codecs.open(self.topic_file, mode='r', encoding='utf-8')):
			if topic_num:
				topic = [0 for i in range(self.D)]
				topic[int(topic_num)] = 1
				topic_assign.append(topic)
		self.doc_num = len(topic_assign)   # M
		print("The number of documents is: " + str(self.doc_num))
		print("The number of topics is: " + str(self.D))
		print("The number of users is: " + str(self.user_num))
		# topic_assign.shape = (self.doc_num, self.D)  # M*D
		self.topic_assign_np = np.array(topic_assign).T  # D*M
		# initialize user interest score: U
		try:
			self.user_interest = np.load(self.rootDir + '100_U_user_interest_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.user_interest = np.ones((self.time_num, self.D, self.user_num))
		try:
			self.user_interest_Uit_hat = np.load(self.rootDir + '100_U_user_interest_hat_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.user_interest_Uit_hat = np.ones((self.time_num, self.D, self.user_num))
		try:
			self.gamma = np.load(self.rootDir + '100_gamma_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.gamma = np.ones(self.user_num)
		try:
			self.eta = np.load(self.rootDir + '100_eta_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.eta = np.ones(self.user_num)

	def Rij_t1(self):
		# Rij_t_dict = dict.fromkeys([i for i in range(20, self.time_num-1)], rij_dict)
		for time in range(20, self.time_num-1):  # t in time_sequence; count from 1
			rij_dict = dict.fromkeys([i for i in range(self.user_num)], [])
			for user_id in range(self.user_num):  # i in (N)
				for item_id in range(self.doc_num):
					print("Processing user " + str(user_id) + " item " + str(item_id) + " at time " + str(time) + "......")
					gamma_i = self.gamma[user_id]
					user = selected_user[user_id]
					Uit_t1 = self.Uit_hat(user, time+1, gamma_i)
					Rij_t1 = np.dot(Uit_t1, self.topic_assign_np[:, item_id])
					rij_dict[user_id].append(Rij_t1)
			print("Writing json file......")
			with codecs.open('Predict_Rij_t' + str(time) + '.json', mode='w') as fo:
				json.dump(rij_dict, fo)

	def Uit_hat(self, user, time, gamma_i):  # user i
		neighbors_i = self.neighbors(user, time-1, 0)
		sum_h = 0.0
		for h in neighbors_i:
			index_h = selected_user.index(h)
			eta_h = self.eta[index_h]
			sum_h += self.L_hit(h, user, time-1, eta_h) * self.U_it(index_h, time-1)
		Uit_hat = (1-gamma_i) * self.U_it(selected_user.index(user), time-1) + gamma_i * sum_h
		self.user_interest_Uit_hat[time][:, selected_user.index(user)] = Uit_hat
		return Uit_hat

	def neighbors(self, user, time, flag):
		# flag = 0 return all neighbors, =1 return only friends.
		neighbors = []  # list of the users who have a link with user_i

		sql = """SELECT `:START_ID`, `:END_ID`  FROM graph_1month_selected WHERE 
			(`:START_ID`<=%s and  `:END_ID` <= %s) and (`:START_ID`=%s or `:END_ID`=%s)  and `build_time` = '%s'""" % (max_uid, max_uid, user, user, time)
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
		if len(friends_i) != 0:
			friends_h = self.neighbors(user_h, time, 1)
			intersec = list(set(friends_i).intersection(set(friends_h)))
			L_hit = eta * is_friend + (1-eta) * float(len(intersec)/len(friends_i))
		else:
			L_hit = 0
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

	def U_it(self, user_id, time):
		# user_id = selected_user.index(user)
		U_it = self.user_interest[time][:, user_id]  # D*1
		return U_it

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-dbpwd", help="Password of database")
	parser.add_argument("-dbIP", help="IP address of database")
	parser.add_argument("-topicFile", help="Topic assignment file")
	parser.add_argument("-mid_dir", help="The dictionary of mid-id map file")
	parser.add_argument("-f", "--feature_dimension", default=50, help="Dimension of features (topic number)")
	parser.add_argument("-u", "--user_num", default=10000, help="Number of users to build subnetwork")
	parser.add_argument("-t", "--time_num", default=30, help="Number of time sequence")
	parser.add_argument("-i", "--iteround", default=1, help="Number of iterations")
	parser.add_argument("-r", "--root_dir", default='./', help="Root dictionary")

	args = parser.parse_args()
	pwd = args.dbpwd
	dbip = args.dbIP
	topic_file = args.topicFile
	mid_dir = args.mid_dir
	feature_dimension = args.feature_dimension
	user_num = args.user_num
	time_num = args.time_num
	iteround = args.iteround
	rootDir = args.root_dir

	IP = itemPrediction(dbip=dbip, dbname='db_weibodata', pwd=pwd, topic_file=topic_file, mid_dir=mid_dir,
	feature_dimension=feature_dimension, user_num=user_num, time_num=time_num, iteround=iteround, rootDir=rootDir)
	IP.Rij_t1()