import argparse
import numpy as np
import pandas as pd
import codecs
import json
import ConnectDB
from selected_user import selected_user
selected_user = selected_user[0:100]
max_uid = selected_user[-1]


class VecSpaces:
	def __init__(self, dbip, dbname, pwd, mid_dir):
		conDB = ConnectDB.ConnectDB(dbip, dbname, pwd)
		self.cursor, self.db = conDB.connect_db()
		self.item_mid_map = np.loadtxt(mid_dir)

	def calculate_act_Rijt(self, time_num=31, user_num=100, item_num=2080):
		"""
		get the actual Rijt into a vector file (numpy file)
		:param time_num: 
		:param user_num: 
		:param item_num: 
		:return: Actual_Rij_t.npy
		"""
		print("Calculating actural rating preference score for each user, each item at different time......")
		# save as np format
		Rijt = np.ones((time_num, user_num, item_num), dtype='int')
		for time in range(0, time_num):
			print("Time: " + str(time))
			for user_id in range(user_num):
				print("User: " + str(user_id))
				for item_id in range(item_num):
					act_rij = self.R_ijt(user_id, item_id, time)
					Rijt[time][user_id][item_id] = act_rij
		print("Saving to dictionary '../data/Actual_Rij_t.npy'......")
		np.save('../data/Actual_Rij_t.npy', Rijt)

		# save as dictionary/ json format
		'''rij_dict = dict.fromkeys([i for i in range(user_num)], [0.0 for i in range(item_num)])
		Rij_t_dict = dict.fromkeys([i for i in range(0, time_num + 1)], rij_dict)
		with codecs.open('Act_Rij_t.json', mode='w') as fo:
			json.dump(rij_dict, fo)'''

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

	def get_friend_type(self, user_num=100, time_num=31):
		"""
		Get the type of relationships between two users at time t 
		:return: friend_type_uijt.npy
		"""
		print("Getting friend type of each pair of users at different time......")
		friend_type = np.zeros((time_num, user_num, user_num))
		for time in range(0, time_num):
			print("Time: " + str(time))
			for user1 in range(user_num):
				print("User: " + str(user1))
				for user2 in range(user1+1, user_num):
					sql = """SELECT * FROM graph_1month_selected 
								WHERE((`:START_ID`=%s AND `:END_ID`=%s ) or (`:START_ID`=%s AND `:END_ID`=%s)) and `build_time` = '%s'""" % (
					user1, user2, user2, user1, time)
					self.cursor.execute(sql)
					ress = self.cursor.fetchall()
					if len(ress) == 0:
						friend_type[time][user1][user2] = 0.0
						friend_type[time][user2][user1] = 0.0
						# return 0
					elif len(ress) == 1:
						friend_type[time][user1][user2] = 0.5
						friend_type[time][user2][user1] = 0.5
					elif len(ress) == 2:
						friend_type[time][user1][user2] = 1.0
						friend_type[time][user2][user1] = 1.0
		print("Saving to dictionary '../data/friend_type_uijt.npy'......")
		np.save('../data/friend_type_uijt.npy', friend_type)

	def get_neighbors(self, user_num=100, time_num=31):
		print("Getting neighbors of each user at different time......")
		# flag = 0 return all neighbors, =1 return only friends.
		ni_follow = dict.fromkeys([i for i in range(user_num)], list)
		follow_dict_flag0 = dict.fromkeys([i for i in range(0, time_num)], ni_follow)
		ni_friend = dict.fromkeys([i for i in range(user_num)], list)
		friend_dict_flag1 = dict.fromkeys([i for i in range(0, time_num)], ni_friend)
		for time in range(0, time_num):
			print("Time: " + str(time))
			for user_id in range(user_num):
				print("User: " + str(user_id))
				user = selected_user(user_id)
				sql = """SELECT `:START_ID`, `:END_ID`  FROM graph_1month_selected WHERE 
					(`:START_ID`<=%s and  `:END_ID` <= %s) and (`:START_ID`=%s or `:END_ID`=%s)  and `build_time` = '%s'""" % (max_uid, max_uid, user, user, time)
				self.cursor.execute(sql)
				results = self.cursor.fetchall()

				follow_dict_flag0[time][user_id] = []
				for res in results:
					user1, user2 = res[0], res[1]
					if user1 == user and user2 not in follow_dict_flag0[time][user_id]:
						follow_dict_flag0[time][user_id].append(user2)
						# follow.append(user2)
					if user2 == user and user1 not in follow_dict_flag0[time][user_id]:
						follow_dict_flag0[time][user_id].append(user1)
						# follow.append(user1)

				follows = []
				followed = []
				for res in results:
					user1, user2 = res[0], res[1]
					if user1 == user:
						follows.append(user2)
					if user2 == user:
						followed.append(user1)
				friends = list(set(follows).intersection(set(followed)))
				friend_dict_flag1[time][user_id] = friends
				# neighbors = friends
		with codecs.open('neighbors_flag_0.json', mode='w') as fo:
			json.dump(follow_dict_flag0, fo)
		with codecs.open('neighbors_flag_1.json', mode='w') as fo:
			json.dump(friend_dict_flag1, fo)


if __name__ == '__main__':
	# print np.ones((2,5,10), dtype='int')

	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--DB_password",  help="Password of database")
	parser.add_argument("-i", "--DB_IP_address", help="IP address of database")
	parser.add_argument("-u", "--users_generate", default='f', help="Choose how to generate selected users, r: randomly generate; f: from file")
	parser.add_argument("-mid_dir", default="../data/mid_id_user100", help="The dictionary of mid-id map file")

	args = parser.parse_args()
	pwd = args.DB_password
	dbip = args.DB_IP_address
	users_flag = args.users_generate
	mid_dir = args.mid_dir
	dbname = 'db_weibodata'

	VS = VecSpaces(dbip, dbname, pwd, mid_dir)
	VS.get_friend_type()
	VS.get_neighbors()
	VS.calculate_act_Rijt()

