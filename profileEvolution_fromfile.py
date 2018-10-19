import argparse
import codecs
import json
import numpy as np
from selected_user import selected_user


class ProfileEvolution:
	def __init__(self, topic_file, learning_rate, max_iter, feature_dimension, user_num, time_num, topic_type, dataDir, outDir):
		self.D = int(feature_dimension)
		# conDB = ConnectDB.ConnectDB(dbip, dbname, pwd)
		# self.cursor, self.db = conDB.connect_db()
		self.topic_file = topic_file  # M*1
		self.user_num = int(user_num)
		self.time_num = int(time_num)
		self.max_iter = int(max_iter)
		# self.minibatch = int(minibatch)
		self.learning_rate = float(learning_rate)
		# self.mid_dir = mid_dir
		# self.item_mid_map = np.loadtxt(self.mid_dir)
		self.topic_type = topic_type
		self.rootDir = dataDir
		self.outDir = outDir
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
		self.doc_num = len(self.topic_assign_np)   # M
		print("Reading Actual_rij_t.npy file......")
		self.R_ij = np.ones((self.time_num, self.user_num, self.doc_num), dtype='int')
		# self.R_ij = np.load(self.rootDir+'Actual_Rij_t.npy')
		print("Reading friend_type_uijt.npy file......")
		self.friend_type = np.zeros((self.time_num, self.user_num, self.user_num))
		# self.friend_type = np.load(self.rootDir + 'friend_type_uijt.npy')
		print("The number of documents is: " + str(self.doc_num))
		print("The number of topics is: " + str(self.D))
		print("The number of users is: " + str(self.user_num))
		# topic_assign.shape = (self.doc_num, self.D)  # M*D
		# initialize user interest score: U
		self.user_interest = np.ones((self.time_num, self.D, self.user_num))
		self.user_interest_Uit_hat = np.ones((self.time_num, self.D, self.user_num))
		self.gamma = np.ones(self.user_num)
		self.eta = np.ones(self.user_num)

	def SGD_Uit(self, lambda_U, round_num):
		print("Processing Uit......")
		for time in range(1, self.time_num-1):  # t in time_sequence; count from 1
			for user_id in range(self.user_num):  # i in (N)
				user = selected_user[user_id]		# get the user id
				print("Processing user " + str(user) + " with id number " + str(user_id) + " at time " + str(time) + "......")
				Uit = self.minimum_Uit(user, time, self.gamma, self.eta, lambda_U)
				self.update_U_it(Uit, user_id, time)
		print("Saving user interest U into file......")
		np.save(self.outDir + self.topic_type + '_U_user_interest_' + str(round_num) + '.npy', self.user_interest)
		np.save(self.outDir + self.topic_type + '_U_user_interest_hat_' + str(round_num) + '.npy', self.user_interest_Uit_hat)
		# for j in self.doc_num(M)

	def minimum_Uit(self, user, time, gamma, eta, lambda_U):  # , item_set
		# user i(id),  time t
		user_h = self.neighbors(user, time, 0)
		user_id = selected_user.index(user)
		gamma_i = gamma[user_id]

		sum_userh = 0.0
		for h in user_h:
			if min(user_h) > selected_user[-1] or selected_user.index(min(user_h)) >= self.user_num:
				break
			uid_h = selected_user.index(h)
			gamma_h = gamma[uid_h]
			eta_h = eta[uid_h]
			sum_userh += gamma_h * self.L_hit(h, user, time, eta_h) * (self.Uit_hat(h, time + 1, gamma_h) - self.U_it(uid_h, time + 1))
		min_Uit = np.zeros(self.D)
		# item_set = []
		# for minb in range(self.minibatch):
		# 	item_set.append(random.randint(0, self.doc_num-1))  # choose mini_batch number of documents' ids
		for iter in range(self.max_iter):
			print("Iteration: " + str(iter))
			for item in range(self.doc_num):
				Y_ijt, R_ijt = self.Y_R_ijt(user, item, time)
				# item j
				uid_time = self.U_it(user_id, time)
				v_j_item = self.V_j(item)
				target_min_Uit = np.add(Y_ijt * (np.dot(uid_time, v_j_item) - R_ijt) *v_j_item, lambda_U*(uid_time-self.Uit_hat(user, time, gamma_i)))
				target_min_Uit = np.add(target_min_Uit, lambda_U*(1-gamma_i)*(self.U_it(user_id, time+1)-self.Uit_hat(user, time+1, gamma_i)))
				target_min_Uit = np.add(target_min_Uit, lambda_U*sum_userh)

			min_Uit = min_Uit - self.learning_rate * target_min_Uit  # )np.add
			if np.isnan(np.sum(min_Uit)):
				return self.user_interest[time][:, user_id]
			self.update_U_it(min_Uit, user_id, time)
			print min_Uit
		return min_Uit

	def PGD_gamma_eta(self, lambda_U, round_num):
		print("Processing gamma and eta......")
		for user_id in range(self.user_num):  # i in (N)
			user = selected_user[user_id]  # get the user
			print("Processing user " + str(user) + " with id number " + str(user_id) + "......")
			gamma_i = self.minimum_gamma(user, lambda_U, self.eta)
			self.gamma[user_id] = gamma_i
			eta_i = self.minimum_eta(user, lambda_U, self.gamma)
			self.eta[user_id] = eta_i

		print("Saving parameter gamma, eta into file......")
		np.save(self.outDir + self.topic_type + '_gamma_' + str(round_num) + '.npy', self.gamma)
		np.save(self.outDir + self.topic_type + '_eta_' + str(round_num) + '.npy', self.eta)

	def minimum_gamma(self, user, lambda_U, eta):
		# user i
		user_id = selected_user.index(user)
		sum_t = 0.0
		gamma_i = 1.0
		for iter in range(self.max_iter):
			print("Iteration: " + str(iter))
			for t in range(1, self.time_num-1):
				neighbors_i = self.neighbors(user, t - 1, 0)
				sum_h = 0.0
				for h in neighbors_i:
					if min(neighbors_i) > selected_user[-1] or selected_user.index(min(neighbors_i)) >= self.user_num:
						break
					uid_h = selected_user.index(h)
					eta_h = eta[uid_h]
					sum_h += self.L_hit(h, user, t - 1, eta_h) * self.user_interest[t-1][:, uid_h]  # self.U_it(h, t - 1)
				# sum_t += (self.Uit_hat(user, t, gamma_i) - self.U_it(user, t)) * (sum_h-self.U_it(user, t - 1))
				sum_t += (
					self.user_interest_Uit_hat[t][:, user_id] - self.user_interest[t][:, user_id] * (sum_h - self.user_interest[t-1][:, user_id]))
			min_target = lambda_U * sum_t
			# gamma_i -= self.learning_rate*min_target
			# gamma_i = self.pi_x(gamma_i-self.learning_rate*min_target)
			inp = gamma_i - min_target
			gamma_i -= self.learning_rate * self.pi_x(inp)
			print "gamma: " + str(gamma_i)
		return gamma_i

	def pi_x(self, x):
		res = []
		for c in range(0, 11):
			c_1 = c * 0.1
			# y = np.array([0.5 for i in range(self.user_num)])
			y = np.array([c_1 - x])
			res.append(np.linalg.norm(y, ord=2))
		# print res
		return min(res)

	def minimum_eta(self, user_i, lambda_U, gamma):
		sum_t = 0.0
		uid_i = selected_user.index(user_i)
		gamma_i = gamma[uid_i]
		eta_i = 1.0
		for iter in range(self.max_iter):
			print("Iteration: " + str(iter))
			for t in range(1, self.time_num-1):
				neighbors_i = self.neighbors(user_i, t - 1, 0)
				sum_h = 0.0
				for user_h in neighbors_i:
					if min(neighbors_i) > selected_user[-1] or selected_user.index(min(neighbors_i)) >= self.user_num:
						break
					uid_h = selected_user.index(user_h)
					is_friend = 0  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends
					friends_i = self.neighbors(user_i, t, 1)
					if len(friends_i) != 0:
						friends_h = self.neighbors(user_h, t, 1)
						intersec = list(set(friends_i).intersection(set(friends_h)))
						sum_h += (is_friend - float(len(intersec) / len(friends_i))) * self.user_interest[t-1][:, uid_h]  # self.U_it(user_h, t-1)
					else:
						sum_h += 0
				# sum_t += (self.Uit_hat(user_i, t, gamma_i) - self.U_it(user_i, t)) * (gamma_i * sum_h + (1-gamma_i) * self.U_it(user_, t-1))
				sum_t += (self.user_interest_Uit_hat[t][:, uid_i] -
						  self.user_interest[t][:, uid_i] * (gamma_i * sum_h + (1 - gamma_i) * self.user_interest[t-1][:, uid_i]))

			min_target = lambda_U * sum_t
			eta_i -= self.learning_rate * self.pi_x(eta_i - min_target)
			# eta_i = self.pi_x(eta_i - self.learning_rate * min_target)
			print "eta: " + str(eta_i)
		return eta_i

	def Y_R_ijt(self, user_i, item_j, time):
		# Y_ijt = 1 if user_i has a link with item_j at time t, else=0
		# R_ijt = rating preference score of user_i to item_j at time t
		# R_ij = np.load(self.rootDir+'Actual_Rij_t.npy')
		usr_id = selected_user.index(user_i)
		R_ijt = self.R_ij[time][usr_id][item_j]
		if R_ijt == 0:
			Y_ijt = 0
		else:
			Y_ijt = 1
		return Y_ijt, R_ijt

	def V_j(self, item):
		"""
		:param item: The id of item
		:return: V_j = topic of item_j
		"""
		V_j = self.topic_assign_np[:, item]  # D*1
		return V_j

	def U_it(self, user_id, time):
		# user_id = selected_user.index(user)
		U_it = self.user_interest[time][:, user_id]  # D*1
		return U_it

	def update_U_it(self, U_it, user_id, time):
		# user_id = selected_user.index(user)
		self.user_interest[time][:, user_id] = U_it  # D*1

	def Uit_hat(self, user, time, gamma_i):  # user i
		user_id = selected_user.index(user)
		neighbors_i = self.neighbors(user, time-1, 0)
		sum_h = 0.0
		for h in neighbors_i:
			if min(neighbors_i) > selected_user[-1] or selected_user.index(min(neighbors_i)) >= self.user_num:
				break
			index_h = selected_user.index(h)
			eta_h = self.eta[index_h]
			sum_h += self.L_hit(h, user, time-1, eta_h) * self.U_it(index_h, time-1)
		Uit_hat = (1-gamma_i) * self.U_it(user_id, time-1) + gamma_i * sum_h
		self.user_interest_Uit_hat[time][:, selected_user.index(user)] = Uit_hat
		return Uit_hat

	def neighbors(self, user, time, flag):
		# user id
		# flag = 0 return all neighbors, =1 return only friends.
		user_id = selected_user.index(user)
		if flag == 0:
			with codecs.open(self.rootDir + 'neighbors_flag_0.json', mode='r') as infile:
				neighbors_0 = json.load(infile)
				try:
					neighbors = [int(i) for i in neighbors_0[str(time)][str(user_id)][1:-1].replace('L','').split(', ')]
				except Exception as e:
					# print e
					neighbors = []
		else:
			with codecs.open(self.rootDir + 'neighbors_flag_1.json', mode='r') as infile:
				neighbors_1 = json.load(infile)
				try:
					neighbors = [int(i) for i in neighbors_1[str(time)][str(user_id)][1:-1].replace('L','').split(', ')]
				except Exception as e:
					# print e
					neighbors = []
		return neighbors

	def L_hit(self, user_h, user_i, time, eta):
		user_h_id = selected_user.index(user_h)
		user_i_id = selected_user.index(user_i)
		is_friend = self.get_friend_type(user_h_id, user_i_id, time)  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends

		friends_i = self.neighbors(user_i, time, 1)
		if len(friends_i) != 0:
			friends_h = self.neighbors(user_h, time, 1)
			intersec = list(set(friends_i).intersection(set(friends_h)))
			L_hit = eta * is_friend + (1-eta) * float(len(intersec)/len(friends_i))
		else:
			L_hit = 0
		return L_hit

	def get_friend_type(self, user1, user2, time):
		# input user id
		# friend_type = np.load(self.rootDir + 'friend_type_uijt.npy')
		try:
			return self.friend_type[time][selected_user.index(user1)][selected_user.index(user2)]
		except Exception as e:
			return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-topicFile", default="../data/topic_assign_user2000", help="Topic assignment file")
	# parser.add_argument("-mid_dir", default="../data/mid_id_user2000", help="The dictionary of mid-id map file")
	parser.add_argument("-l", "--learning_rate", default=0.001, help="The learning rate of SGD for Uit")
	# parser.add_argument("-b", "--minibatch", default=1000, help="Number of minibatch of SGD (subset of documents)")
	parser.add_argument("-i", "--max_iteration", default=1000, help="The max iteration of SGD")
	parser.add_argument("-f", "--feature_dimension", default=100, help="Dimension of features (topic number)")
	parser.add_argument("-u", "--user_num", default=2000, help="Number of users to build subnetwork")
	parser.add_argument("-t", "--time_num", default=31, help="Number of time sequence")
	parser.add_argument("-tt", "--topic_type", default='DMM', help="Topic model type, LDA or DMM")
	parser.add_argument("-r", "--rootDir", default='../data/', help="Root data dictionary")
	parser.add_argument("-o", "--outDir", default='../output/', help="Output dictionary")

	args = parser.parse_args()
	topic_file = args.topicFile
	# mid_dir = args.mid_dir
	learning_rate = args.learning_rate
	max_iteration = args.max_iteration
	feature_dimension = args.feature_dimension
	user_num = args.user_num
	time_num = args.time_num
	topic_type = args.topic_type
	rootDir = args.rootDir
	outDir = args.outDir

	selected_user = selected_user[0:user_num]
	# topic_file = 'E:\\code\\SN2\\pDMM-master\\output\\model.filter.sense.topicAssignments'
	# mid_dir = 'E:\\data\\social netowrks\\weibodata\\processed\\root_content_id.txt'
	Profile = ProfileEvolution(topic_file=topic_file,
							   learning_rate=learning_rate, max_iter=max_iteration,
							   feature_dimension=feature_dimension, user_num=user_num, time_num=time_num,
							   topic_type=topic_type, dataDir=rootDir, outDir=outDir)
	# gamma = np.array([0.5 for i in range(user_num)])
	# eta = np.array([0.5 for i in range(user_num)])
	lambda_U = 0.3
	for i in range(10):
		print(str(i) + "-th round......")
		Profile.SGD_Uit(lambda_U, i)
		Profile.PGD_gamma_eta(lambda_U, i)
	# Profile.Y_R_ijt(1227898, 3361644068075147, '2011-11-02-11:18:14')
