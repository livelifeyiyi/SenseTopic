import argparse
import codecs
import numpy as np
import ConnectDB
import random
from selected_user import selected_user


class ProfileEvolution:
	def __init__(self, dbip, dbname, pwd, topic_file, mid_dir, learning_rate, minibatch, max_iter, feature_dimension, user_num, time_num):
		self.D = int(feature_dimension)
		conDB = ConnectDB.ConnectDB(dbip, dbname, pwd)
		self.cursor, self.db = conDB.connect_db()
		self.topic_file = topic_file  # M*1
		self.user_num = int(user_num)
		self.time_num = int(time_num)
		self.max_iter = int(max_iter)
		self.minibatch = int(minibatch)
		self.learning_rate = float(learning_rate)
		self.mid_dir = mid_dir
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
		self.user_interest = np.ones((self.time_num, self.D, self.user_num))
		self.user_interest_Uit_hat = np.ones((self.time_num, self.D, self.user_num))
		self.gamma = np.ones(self.user_num)
		self.eta = np.ones(self.user_num)

	def SGD_Uit(self, lambda_U, round_num):
		for time in range(1, self.time_num-1):  # t in time_sequence; count from 1
			for user_id in range(10):  # self.user_num):  # i in (N)
				user = selected_user[user_id]		# get the user id
				print("Processing user " + str(user) + " with id number " + str(user_id) + " at time " + str(time) + "......")
				Uit = self.minimum_Uit(user, time, self.gamma, self.eta, lambda_U)
				self.update_U_it(Uit, user_id, time)
		print("Saving user interest U into file......")
		np.save('../output/U_user_interest_' + str(round_num) + '.npy', self.user_interest)
		np.save('../output/U_user_interest_hat_' + str(round_num) + '.npy', self.user_interest_Uit_hat)
		# for j in self.doc_num(M)

	def minimum_Uit(self, user, time, gamma, eta, lambda_U):  # , item_set
		# user i(id),  time t
		user_h = self.neighbors(user, time, 0)
		user_id = selected_user.index(user)
		gamma_i = gamma[user_id]

		sum_userh = 0.0
		for h in user_h:
			uid_h = selected_user.index(h)
			gamma_h = gamma[uid_h]
			eta_h = eta[uid_h]
			sum_userh += gamma_h * self.L_hit(h, user, time, eta_h) * (self.Uit_hat(h, time + 1, gamma_h) - self.U_it(uid_h, time + 1))
		min_Uit = np.zeros(self.D)
		item_set = []
		for minb in range(self.minibatch):
			item_set.append(random.randint(0, self.doc_num-1))  # choose mini_batch number of documents' ids
		for iter in range(self.max_iter):
			print("Iteration: " + str(iter))
			for item in item_set:
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
		for user_id in range(self.user_num):  # i in (N)
			user = selected_user[user_id]  # get the user
			print("Processing user " + str(user) + " with id number " + str(user_id) + "......")
			gamma_i = self.minimum_gamma(user, lambda_U, self.eta)
			self.gamma[user_id] = gamma_i
			eta_i = self.minimum_eta(user, lambda_U, self.gamma)
			self.eta[user_id] = eta_i

		print("Saving parameter gamma, eta into file......")
		np.save('../output/gamma_' + str(round_num) + '.npy', self.gamma)
		np.save('../output/eta_' + str(round_num) + '.npy', self.eta)

	def minimum_gamma(self, user, lambda_U, eta):
		# user i
		user_id = selected_user.index(user)
		sum_t = 0.0
		gamma_i = 1.0
		for iter in range(self.max_iter):
			print("Iteration: " + str(iter))
			for t in range(1, self.time_num+1):
				neighbors_i = self.neighbors(user, t - 1, 0)
				sum_h = 0.0
				for h in neighbors_i:
					uid_h = selected_user.index(h)
					eta_h = eta[uid_h]
					sum_h += self.L_hit(h, user, t - 1, eta_h) * self.user_interest[t-1][:, uid_h]  # self.U_it(h, t - 1)
				# sum_t += (self.Uit_hat(user, t, gamma_i) - self.U_it(user, t)) * (sum_h-self.U_it(user, t - 1))
				sum_t += (
					self.user_interest_Uit_hat[t][:, user_id] - self.user_interest[t][:, user_id] * (sum_h - self.user_interest[t-1][:, user_id]))
			min_target = lambda_U * sum_t
			# gamma_i -= self.learning_rate*min_target
			gamma_i = self.pi_x(gamma_i-self.learning_rate*min_target)
			print "gamma: " + str(gamma_i)
		return gamma_i

	def pi_x(self, x):
		res = []
		for c in range(0, 1.1, 0.1):
			y = np.array([0.5 for c in range(self.user_num)])
			res.append(np.linalg.norm(y-x, ord=2))
		return np.argmin(np.array(res))

	def minimum_eta(self, user_i, lambda_U, gamma):
		sum_t = 0.0
		gamma_i = gamma[user_i]
		uid_i = selected_user.index(user_i)
		eta_i = 1.0
		for iter in range(self.max_iter):
			print("Iteration: " + str(iter))
			for t in range(1, self.time_num+1):
				neighbors_i = self.neighbors(user_i, t - 1, 0)
				sum_h = 0.0
				for user_h in neighbors_i:
					uid_h = selected_user.index(user_h)
					is_friend = 0  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends
					friends_i = self.neighbors(user_i, t, 1)
					friends_h = self.neighbors(user_h, t, 1)
					intersec = list(set(friends_i).intersection(set(friends_h)))
					sum_h += (is_friend - float(len(intersec) / len(friends_i))) * self.user_interest[t-1][:, uid_h]  # self.U_it(user_h, t-1)

				# sum_t += (self.Uit_hat(user_i, t, gamma_i) - self.U_it(user_i, t)) * (gamma_i * sum_h + (1-gamma_i) * self.U_it(user_, t-1))
				sum_t += (self.user_interest_Uit_hat[t][:, uid_i] -
						  self.user_interest[t][:, uid_i] * (gamma_i * sum_h + (1 - gamma_i) * self.user_interest[t-1][:, uid_i]))

			min_target = lambda_U * sum_t
			eta_i = self.pi_x(eta_i - self.learning_rate * min_target)
			print "eta: " + str(eta_i)
		return eta_i

	def Y_R_ijt(self, user_i, item_j, time):
		# Y_ijt = 1 if user_i has a link with item_j at time t, else=0
		# R_ijt = rating preference score of user_i to item_j at time t
		# `type` tb_miduserrelation_selected  # SELECT * FROM tb_miduserrelation
		mid = self.item_mid_map[item_j]
		sql = """SELECT `type` FROM tb_miduserrelation_selected_time
				WHERE `:START_ID`=%s AND `:END_ID`=%s AND `time_index`="%s" """ % (user_i, mid, time)
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

	def U_it(self, user_id, time):
		# user_id = selected_user.index(user)
		U_it = self.user_interest[time][:, user_id]  # D*1
		return U_it

	def update_U_it(self, U_it, user_id, time):
		# user_id = selected_user.index(user)
		self.user_interest[time][:, user_id] = U_it  # D*1

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-dbpwd", help="Password of database")
	parser.add_argument("-dbIP", help="IP address of database")
	parser.add_argument("-topicFile", help="Topic assignment file")
	parser.add_argument("-mid_dir", help="The dictionary of mid-id map file")
	parser.add_argument("-l", "--learning_rate", default=0.001, help="The learning rate of SGD for Uit")
	parser.add_argument("-b", "--minibatch", default=1000, help="Number of minibatch of SGD (subset of documents)")
	parser.add_argument("-i", "--max_iteration", default=1000, help="The max iteration of SGD")
	parser.add_argument("-f", "--feature_dimension", default=50, help="Dimension of features (topic number)")
	parser.add_argument("-u", "--user_num", default=10000, help="Number of users to build subnetwork")
	parser.add_argument("-t", "--time_num", default=30, help="Number of time sequence")

	args = parser.parse_args()
	pwd = args.dbpwd
	dbip = args.dbIP
	topic_file = args.topicFile
	mid_dir = args.mid_dir
	learning_rate = args.learning_rate
	minibatch = args.minibatch
	max_iteration = args.max_iteration
	feature_dimension = args.feature_dimension
	user_num = args.user_num
	time_num = args.time_num

	# topic_file = 'E:\\code\\SN2\\pDMM-master\\output\\model.filter.sense.topicAssignments'
	# mid_dir = 'E:\\data\\social netowrks\\weibodata\\processed\\root_content_id.txt'
	Profile = ProfileEvolution(dbip=dbip, dbname='db_weibodata', pwd=pwd, topic_file=topic_file, mid_dir=mid_dir,
							   learning_rate=learning_rate, minibatch=minibatch, max_iter=max_iteration,
							   feature_dimension=feature_dimension, user_num=user_num, time_num=time_num)
	# gamma = np.array([0.5 for i in range(user_num)])
	# eta = np.array([0.5 for i in range(user_num)])
	lambda_U = 0.3
	for i in range(2):
		print(str(i) + "-th round......")
		Profile.SGD_Uit(lambda_U, i)
		Profile.PGD_gamma_eta(lambda_U, i)
	# Profile.Y_R_ijt(1227898, 3361644068075147, '2011-11-02-11:18:14')
