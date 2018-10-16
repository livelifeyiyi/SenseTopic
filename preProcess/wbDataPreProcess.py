import MySQLdb
import codecs
import pandas as pd
import random
import argparse
import SenseTopic.ConnectDB
import time
import numpy as np
import json


class SelectUser:
	def __init__(self, select_user_num, all_user_num):
		"""
		
		:param select_user_num: number of users need to be selected
		:param all_user_num: total number of users
		"""
		self.select_user_num = select_user_num
		self.all_user_num = all_user_num

	def random_select_users(self):
		"""
		randomly select some of the users to build a sub graph. 	
		:return: selected user list.
		"""
		select_user =[]
		for i in range(self.select_user_num):
			select_user.append(random.randint(0, self.all_user_num))
		select_user = sorted(select_user)
		return select_user


class BuildSubNetwork:
	def __init__(self, selected_user, dbip, dbname, pwd, mid_dir):
		self.selected_user = selected_user
		conDB = SenseTopic.ConnectDB.ConnectDB(dbip, dbname, pwd)
		self.cursor, self.db = conDB.connect_db()
		# self.item_mid_map = np.loadtxt(mid_dir)

	def graph_1month_select(self):
		print("Choosing the randomly selected users and time from 170w users......")
		with codecs.open("../graph_170w_1month.txt", mode='r', encoding='utf8') as graphfile:  # E:/data/social netowrks/weibodata
			# with codecs.open("graph_1month_selecteduser.txt", mode='a+', encoding='utf-8') as outfile:
			line = graphfile.readline()
			i = 0
			while line:
				'''if i < 3818593:
					line = graphfile.readline()
					i += 1
					continue'''
				l = line.strip('\n').strip('\r').split()
				id1, id2, time = l[0], l[1], l[2]
				'''sql = """INSERT INTO graph_170w_1month (`:START_ID`,`:END_ID`, `build_time`) 
					VALUES (%s, %s, %s) """ % (id1, id2, time)
				self.cursor.execute(sql)
				self.db.commit()'''
				if (int(id1) in self.selected_user) and (int(id2) in self.selected_user):
					# outfile.write(id1 + '\t' + id2 + '\t' + time + '\n')
					sql = """INSERT INTO graph_1month_selected (`:START_ID`,`:END_ID`, `build_time`) 
											VALUES (%s, %s, %s) """ % (id1, id2, time)
					self.cursor.execute(sql)
					self.db.commit()
				line = graphfile.readline()
				if i % 10000 == 0:
					print str(i) + '-th line finished.'
				i += 1
		print("User selection done! Results saved into table graph_1month_selected")

	def user_relation_select(self):
		print("Choosing the randomly selected users' network from tb_userrelation......")
		in_p = str(self.selected_user)[1:-1]  # ', '.join(self.selected_user)
		sql = """ CREATE table tb_userrelation_selected 
				(SELECT `:START_ID`,`:END_ID`,`TYPE` 
				FROM tb_userrelation WHERE `:START_ID` IN (%s) AND `:END_ID` IN (%s) )""" % (in_p, in_p)
		self.cursor.execute(sql)
		self.db.commit()

	def user_mid_select(self):
		print("Choosing the randomly selected users and mid from tb_miduserrelation......")
		in_p = str(self.selected_user)[1:-1]
		sql = """ CREATE table tb_miduserrelation_selected 
						(SELECT `:START_ID`, `type`, `time`, `:END_ID` 
						FROM tb_miduserrelation WHERE `:START_ID` IN (%s) )""" % (in_p)
		self.cursor.execute(sql)
		self.db.commit()

	def convert_time_format(self, dbip, pwd, dbname):
		"""
		covert the String format time in table 'tb_miduserrelation_selected' into time index format 
		:return: 
		"""
		import time
		from sqlalchemy import create_engine
		engine = create_engine('mysql://root:%s=@%s/%s?charset=utf8'%(pwd, dbip, dbname))
		# print time.mktime(time.strptime('2012-09-29 00:00:00', '%Y-%m-%d %H:%M:%S'))
		init_timestamp = time.mktime(time.strptime('2012-09-29', '%Y-%m-%d'))
		table = pd.read_sql_table('tb_miduserrelation_selected', engine)
		time_sequence = table.ix[:, 'time']
		time_index = []
		i = 0
		for each_time in time_sequence:
			print i
			i += 1
			current_timestamp = time.mktime(time.strptime(each_time[0:10], '%Y-%m-%d'))
			# print each_time, current_timestamp
			if current_timestamp < init_timestamp:
				time_index.append(0)
			else:
				time_index.append(((current_timestamp - init_timestamp) / 86400) + 1)
		table.insert(4, 'time_index', time_index, allow_duplicates=True)
		table.to_sql('tb_miduserrelation_selected_time', engine, if_exists='replace', index=False)

		print("Table replacement finished!")

	def mid_selected_user(self, usernum):
		"""
		selected the user concerned mid
		:return: save to file "mid_id_user100"
		"""
		from SenseTopic.selected_user import selected_user
		max_user = selected_user[usernum-1]
		sql = """ select distinct(`:END_ID`) from tb_miduserrelation_selected_time where `:START_ID` <= %s """% max_user
		self.cursor.execute(sql)
		ress = self.cursor.fetchall()
		for res in ress:
			mid = res[0]
			with codecs.open('../data/mid_id_user'+str(usernum), mode='a+', encoding='utf-8') as mid_file:
				mid_file.write(str(mid) + '\n')

	def topic_assign_user(self, usernum):
		"""
		Choose the topic assignments for selected users's mid
		:return: save to file "topic_assign_user100"
		"""
		topic_assign_file = "E:/code/SN2/pDMM-master/output/model.filterAllstop.sense.100.topicAssignments"
		mid_id_all_file = "../data/root_content_id.txt"
		mid_id_user100 = "../data/mid_id_user" + str(usernum)  # mid_id_user100
		target_file = "../data/topic_assign_user" + str(usernum) # topic_assign_user100
		item_mid_map = list(np.loadtxt(mid_id_all_file))
		mid_id_user100_np = np.loadtxt(mid_id_user100)
		topic_assign_np = np.loadtxt(topic_assign_file)
		for mid_user100 in mid_id_user100_np:
			mid_id = item_mid_map.index(mid_user100)
			selected_topic_assign = int(topic_assign_np[mid_id])
			with codecs.open(target_file, mode='a+', encoding='utf-8') as outfile:
				outfile.write(str(selected_topic_assign) + '\n')

	def process_topoc100_assign(self, user_num):
		from topicDistribution import topic
		from midIndex import mid

		mid_id_user100 = "mid_id_user" + str(user_num)  # mid_id_user100
		target_file = "topic_assign_user"+str(user_num)+"_lda"  # topic_assign_user100
		# with codecs.open(topic_assign_file, mode='r', encoding='utf-8') as topic_file:
		# 	topic_assign_np = np.array(list(topic_file.readlines()))
		topic_assign_np = np.array(topic)
		# with codecs.open(mid_id_all_file, mode='r', encoding='utf-8') as mid_file:
		# 	item_mid_map = list(mid_file.readlines())
		mid_id_user100_np = np.loadtxt(mid_id_user100)
		for mid_user100 in mid_id_user100_np:
			try:
				mid_id = mid.index(str(int(mid_user100)))
				selected_topic_assign = topic_assign_np[mid_id]
			except Exception as e:
				print e
				selected_topic_assign = [str(0.0) for i in range(100)]
				selected_topic_assign = ' '.join(selected_topic_assign)
			with codecs.open(target_file, mode='a+', encoding='utf-8') as outfile:
				outfile.write(str(selected_topic_assign) + '\n')

	def mid_selected_user_1w(self):
		sql = """ select distinct(`:END_ID`) from tb_miduserrelation_selected_time"""
		self.cursor.execute(sql)
		ress = self.cursor.fetchall()
		for res in ress:
			mid = res[0]
			with codecs.open('../data/mid_id_user1w', mode='a+', encoding='utf-8') as mid_file:
				mid_file.write(str(mid) + '\n')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--DB_password",  help="Password of database")
	parser.add_argument("-i", "--DB_IP_address", help="IP address of database")
	parser.add_argument("-u", "--users_generate", default='f', help="Choose how to generate selected users, r: randomly generate; f: from file")
	args = parser.parse_args()
	pwd = args.DB_password
	dbip = args.DB_IP_address
	users_flag = args.users_generate
	dbname = 'db_weibodata'
	# randomly generate selected user
	if users_flag == 'r':
		SelectUser = SelectUser(select_user_num=10000, all_user_num=1787443)
		selected_user = SelectUser.random_select_users()
		with codecs.open('../selected_user_' + str(time.time()) + '.txt', mode='w', encoding='utf-8') as outfile:
			outfile.write(str(selected_user))

	# include selected user from file
	else:
		from SenseTopic.selected_user import selected_user

	BuildSubNetwork = BuildSubNetwork(selected_user, dbip=dbip, dbname=dbname, pwd=pwd, mid_dir='../data/mid_id_user100')
	# BuildSubNetwork.graph_1month_select()
	# BuildSubNetwork.user_relation_select()
	# BuildSubNetwork.user_mid_select()
	# BuildSubNetwork.convert_time_format(dbip, pwd, dbname)
	# BuildSubNetwork.mid_selected_user(2000)
	# BuildSubNetwork.mid_selected_user_1w()
	# BuildSubNetwork.topic_assign_user(2000)
	BuildSubNetwork.process_topoc100_assign(2000)
	# from selected_user import selected_user
	# print selected_user[0:2000]