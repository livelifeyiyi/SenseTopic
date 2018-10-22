import argparse

import numpy as np
import ConnectDB
import codecs
import json
from selected_user import selected_user
# selected_user = selected_user[0:100]
# max_uid = selected_user[-1]


class itemPrediction:
	def __init__(self, dbip, dbname, pwd, topic_file, mid_dir, feature_dimension, user_num, time_num, iteround, rootDir, topic_type):
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
		self.topic_type = topic_type
		print("Reading the topic assignment file......")
		if self.topic_type == 'DMM':
			topic_assign = []
			for doc_id, topic_num in enumerate(codecs.open(self.topic_file, mode='r', encoding='utf-8')):
				if topic_num:
					topic = [0 for i in range(self.D)]
					topic[int(topic_num)] = 1
					topic_assign.append(topic)
			self.topic_assign_np = np.array(topic_assign).T  # D*M
		else:
			self.topic_assign_np = np.loadtxt(self.topic_file).T
		self.doc_num = len(topic_assign)   # M
		print("The number of documents is: " + str(self.doc_num))
		print("The number of topics is: " + str(self.D))
		print("The number of users is: " + str(self.user_num))
		# topic_assign.shape = (self.doc_num, self.D)  # M*D
		# initialize user interest score: U
		try:
			self.user_interest = np.load(self.rootDir + self.topic_type + '_100_U_user_interest_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.user_interest = np.ones((self.time_num, self.D, self.user_num))
		try:
			self.user_interest_Uit_hat = np.load(self.rootDir + self.topic_type + '_100_U_user_interest_hat_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.user_interest_Uit_hat = np.ones((self.time_num, self.D, self.user_num))
		try:
			self.gamma = np.load(self.rootDir + self.topic_type + '_100_gamma_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.gamma = np.ones(self.user_num)
		try:
			self.eta = np.load(self.rootDir + self.topic_type + '_100_eta_' + self.iteround + '.npy')
		except Exception as e:
			print e
			self.eta = np.ones(self.user_num)

	def Rij_t1(self):
		# Rij_t_dict = dict.fromkeys([i for i in range(20, self.time_num-1)], rij_dict)
		Rijt = np.ones((self.time_num, self.user_num, self.doc_num), dtype='int')
		for time in range(1, self.time_num-1):  # t in time_sequence; count from 1
			# rij_dict = dict.fromkeys([i for i in range(self.user_num)], [])
			for user_id in range(self.user_num):  # i in (N)
				for item_id in range(self.doc_num):
					print("Processing user " + str(user_id) + " item " + str(item_id) + " at time " + str(time) + "......")
					gamma_i = self.gamma[user_id]
					user = selected_user[user_id]
					Uit_t1 = self.Uit_hat(user, time+1, gamma_i)
					Rij_t1 = np.dot(Uit_t1, self.topic_assign_np[:, item_id])
					Rijt[time][user_id][item_id] = Rij_t1
					# rij_dict[user_id].append(Rij_t1)
			print("Writing np file......")
			np.save('Predict_Rij_t' + str(time) + '.npy', Rijt[time])
			# with codecs.open('Predict_Rij_t' + str(time) + '.json', mode='w') as fo:
			# 	json.dump(rij_dict, fo)

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
		# neighbors = []  # list of the users who have a link with user_i
		if flag == 0:
			with codecs.open(self.rootDir + 'neighbors_flag_0.json', mode='r') as infile:
				neighbors_0 = json.load(infile)
				neighbors = neighbors_0[time][user]
		else:
			with codecs.open(self.rootDir + 'neighbors_flag_1.json', mode='r') as infile:
				neighbors_1 = json.load(infile)
				neighbors = neighbors_1[time][user]
		return neighbors

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
		friend_type = np.load(self.rootDir + 'friend_type_uijt.npy')
		return friend_type[time][selected_user.index(user1)][selected_user.index(user2)]

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
	parser.add_argument("-u", "--user_num", default=2000, help="Number of users to build subnetwork")
	parser.add_argument("-t", "--time_num", default=31, help="Number of time sequence")
	parser.add_argument("-i", "--iteround", default=1, help="Number of iterations")
	parser.add_argument("-r", "--root_dir", default='./', help="Root dictionary")
	parser.add_argument("-tt", "--topic_type", default='DMM', help="Root dictionary")

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
	topic_type = args.topic_type

	IP = itemPrediction(dbip=dbip, dbname='db_weibodata', pwd=pwd, topic_file=topic_file, mid_dir=mid_dir,
	feature_dimension=feature_dimension, user_num=user_num, time_num=time_num, iteround=iteround, rootDir=rootDir, topic_type=topic_type)
	IP.Rij_t1()