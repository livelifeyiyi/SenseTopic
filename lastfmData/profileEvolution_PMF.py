# -*- coding: utf-8 -*-
import codecs

import json
import numpy as np
from selected_user import selected_user


class PMF(object):
	def __init__(self, topic_file, topic_type, time_num, rootDir, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=1000, num_batches=20, batch_size=1000):
		self.num_feat = num_feat  # Number of latent features,
		self.epsilon = epsilon  # learning rate,
		self._lambda = _lambda  # L2 regularization,
		self.momentum = momentum  # momentum of the gradient,
		self.maxepoch = maxepoch  # Number of epoch before stop,
		self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
		self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)
		# self.user_num = user_num
		self.w_Item = None  # Item feature vectors
		self.w_User = None  # User feature vectors
		self.w_User_hat = None

		self.gamma = None
		self.eta = None

		self.rmse_train = []
		self.rmse_test = []

		self.topic_file = topic_file  # M*1
		self.topic_type = topic_type
		self.time_num = time_num

		print("Reading neighbors_flag_0.json file......")
		with codecs.open(rootDir + 'neighbors_flag_0.json', mode='r') as infile:
			self.neighbors_01 = json.load(infile)
		print("Reading friend_type_uijt.npy file......")
		# self.friend_type = np.zeros((self.time_num, self.user_num, self.user_num))
		self.friend_type = np.load(rootDir + 'friend_type_uijt.npy')

	# ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
	# ***************** train_vec=TrainData, test_vec=TestData*************#
	def fit(self, train_vec, test_vec):
		# mean subtraction
		self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值

		pairs_train = train_vec.shape[0]  # traindata 中条目数
		pairs_test = test_vec.shape[0]  # testdata中条目数

		# 1-p-i, 2-m-c
		num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  # 第0列，user总数
		num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # 第1列，movie总数
		# num_user = 1892
		# num_item = 12316

		incremental = False  # 增量
		if (not incremental) or (self.w_Item is None):
			# initialize
			self.epoch = 0
			if self.topic_type == 'DMM':
				topic_assign = []
				for doc_id, topic_num in enumerate(codecs.open(self.topic_file, mode='r', encoding='utf-8')):
					if topic_num:
						topic = [0 for i in range(self.D)]
						topic[int(topic_num)] = 1
						topic_assign.append(topic)
				self.w_Item = np.array(topic_assign)  # .T  # D*M
			else:
				self.w_Item = np.loadtxt(self.topic_file)  # .T
			# self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
			# self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵
			self.w_User = 0.1 * np.random.randn(self.time_num, num_user,
												self.num_feat)  # numpy.random.randn 用户 T x N x D 正态分布矩阵
			self.w_User_hat = 0.1 * np.random.randn(self.time_num, num_user,
													self.num_feat)  # numpy.random.randn 用户 T x N x D 正态分布矩阵

			# self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
			self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

			self.gamma = (np.random.randint(10, size=num_user)) * 0.1
			self.eta = (np.random.randint(10, size=num_user)) * 0.1

		while self.epoch < self.maxepoch:  # 检查迭代次数
			self.epoch += 1
			if self.epoch % 200 == 0:
				np.save("w_user_%s.npy" % self.epoch, self.w_User)
				np.save("w_user_hat_%s.npy" % self.epoch, self.w_User_hat)
			# Shuffle training truples
			shuffled_order = np.arange(train_vec.shape[0])  # 根据记录数创建等差array
			np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

			# Batch update
			for time_id in range(0, self.time_num - 1):
				for batch in range(self.num_batches):  # 每次迭代要使用的数据量

					print("Time: %s, batch: %s" % (time_id, batch))
					# print "epoch %d batch %d" % (self.epoch, batch+1)

					test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
					batch_idx = np.mod(test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标

					batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
					batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

					# Compute Objective Function
					# ?????
					pred_out = np.sum(np.multiply(self.w_User[time_id][batch_UserID, :],
												  self.w_Item[batch_ItemID, :]),
									  axis=1)  # mean_inv subtracted # np.multiply对应位置元素相乘

					rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

					# print (1 - self.eta[batch_UserID])
					# print rawErr
					# print self.sum_userh(batch_UserID, time_id)
					# Compute gradients
					# ?????
					# print self.sum_userh(batch_UserID, time_id)
					Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
							  + self._lambda * (self.w_User[time_id][batch_UserID, :] - self.Uit_hat(batch_UserID, time_id)) \
							  + self._lambda * np.multiply((1- self.eta[batch_UserID])[:, np.newaxis], (self.w_User[time_id+1][batch_UserID, :] - self.Uit_hat(batch_UserID, time_id+1))) \
							  + self._lambda * self.sum_userh(batch_UserID, time_id)

					# Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[time_id][batch_UserID, :]) \
					# 		  + self._lambda * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

					# dw_Item = np.zeros((num_item, self.num_feat))
					dw_User = np.zeros((num_user, self.num_feat))

					# loop to aggreate the gradients of the same element
					for i in range(self.batch_size):
						# dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
						dw_User[batch_UserID[i], :] += Ix_User[i, :]

					# Update with momentum
					# self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
					self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

					# self.w_Item = self.w_Item - self.w_Item_inc
					self.w_User[time_id] = self.w_User[time_id] - self.w_User_inc

					# Compute Objective Function after
					if batch == self.num_batches - 1:
						pred_out = np.sum(np.multiply(self.w_User[time_id][np.array(train_vec[:, 0], dtype='int32'), :],
													  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
										  axis=1)  # mean_inv subtracted
						rawErr = pred_out - train_vec[:, 2] + self.mean_inv
						obj = np.linalg.norm(rawErr) ** 2 \
							  + 0.5 * self._lambda * (np.linalg.norm(self.w_User[time_id]) ** 2 + np.linalg.norm(self.w_Item) ** 2)

						self.rmse_train.append(np.sqrt(obj / pairs_train))

					# Compute validation error
					if batch == self.num_batches - 1:
						pred_out = np.sum(np.multiply(self.w_User[time_id][np.array(test_vec[:, 0], dtype='int32'), :],
													  self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]),
										  axis=1)  # mean_inv subtracted
						rawErr = pred_out - test_vec[:, 2] + self.mean_inv
						self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

						# Print info
						if batch == self.num_batches - 1:
							print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))

	def sum_userh(self, batch_UserID, time_id):
		sum = []
		for uid in batch_UserID:
			sum_userh = [0.0 for i in range(self.num_feat)]
			user_h = self.neighbors(uid, time_id, 0)
			for h in user_h:
				if h > selected_user[-1]:
					break
				uid_h = selected_user.index(h)
				gamma_h = self.gamma[uid_h]
				eta_h = self.eta[uid_h]
				sum_userh += gamma_h * self.L_hit(uid_h, uid, time_id, eta_h) * (self.Uit_hat([uid_h], time_id + 1) - self.w_User[time_id+1][uid_h, :])
			sum.append(sum_userh)
		return np.array(sum)  # np.array(sum).reshape(self.batch_size, self.num_feat)

	def Uit_hat(self, uid_list, time_id):  # user i
		Uit_hat = []
		for uid in uid_list:
			gamma_i = self.gamma[uid]
			# user_id = selected_user.index(uid)
			neighbors_i = self.neighbors(uid, time_id-1, 0)
			sum_h = 0.0
			for h in neighbors_i:
				if h > selected_user[-1]:
					break
				index_h = selected_user.index(h)
				eta_h = self.eta[index_h]
				sum_h += self.L_hit(index_h, uid, time_id-1, eta_h) * self.w_User[time_id-1][index_h, :]  # self.U_it(index_h, time_id-1)
			uit_each = (1-gamma_i) * self.w_User[time_id-1][uid, :] + gamma_i * sum_h
			if len(uid_list) == 1:
				Uit_hat = uit_each
			else:
				Uit_hat.append(uit_each)  # self.U_it(user_id, time_id-1)
			self.w_User_hat[time_id][uid, :] = uit_each
		return np.array(Uit_hat)  # np.array(Uit_hat)

	def neighbors(self, user_id, time, flag):
		# user id
		# flag = 0 return all neighbors, =1 return only friends.
		if flag == 0:
				try:
					neighbors = [int(i) for i in self.neighbors_01[str(time)][str(user_id)][1:-1].replace('L','').split(', ')]
				except Exception as e:
					# print e
					neighbors = []
		else:
				try:
					neighbors = [int(i) for i in self.neighbors_01[str(time)][str(user_id)][1:-1].replace('L','').split(', ')]
				except Exception as e:
					# print e
					neighbors = []
		return neighbors

	def L_hit(self, user_h_id, user_i_id, time, eta):
		is_friend = self.get_friend_type(user_h_id, user_i_id, time)  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends

		friends_i = self.neighbors(user_i_id, time, 1)
		if len(friends_i) != 0:
			friends_h = self.neighbors(user_h_id, time, 1)
			intersec = list(set(friends_i).intersection(set(friends_h)))
			L_hit = eta * is_friend + (1-eta) * float(len(intersec)/float(len(friends_i)))
		else:
			L_hit = 0.0
		return L_hit

	def get_friend_type(self, user1_id, user2_id, time):
		# input user id
		# friend_type = np.load(self.rootDir + 'friend_type_uijt.npy')
		time = 0
		try:
			return self.friend_type[time][user1_id][user2_id]
		except Exception as e:
			return 0.0

	def predict(self, invID):
		return np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot 点乘

	# ****************Set parameters by providing a parameter dictionary.  ***********#
	def set_params(self, parameters):
		if isinstance(parameters, dict):
			self.num_feat = parameters.get("num_feat", 10)
			self.epsilon = parameters.get("epsilon", 1)
			self._lambda = parameters.get("_lambda", 0.1)
			self.momentum = parameters.get("momentum", 0.8)
			self.maxepoch = parameters.get("maxepoch", 20)
			self.num_batches = parameters.get("num_batches", 10)
			self.batch_size = parameters.get("batch_size", 1000)
			self.topic_file = parameters.get("topic_file", "../data/topic_assign_user2000")
			self.topic_type = parameters.get("topic_type", "LSTM")
			self.time_num = parameters.get("time_num", 8)
			self.rootDir = parameters.get("rootDir", "../data/")
			# topic_file, topic_type, time_num, rootDir

	def topK(self, test_vec, k=10):
		inv_lst = np.unique(test_vec[:, 0])
		pred = {}
		for inv in inv_lst:
			if pred.get(inv, None) is None:
				pred[inv] = np.argsort(self.predict(inv))[-k:]  # numpy.argsort索引排序

		intersection_cnt = {}
		for i in range(test_vec.shape[0]):
			if test_vec[i, 1] in pred[test_vec[i, 0]]:
				intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
		invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

		precision_acc = 0.0
		recall_acc = 0.0
		for inv in inv_lst:
			precision_acc += intersection_cnt.get(inv, 0) / float(k)
			recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

		return precision_acc / len(inv_lst), recall_acc / len(inv_lst)
