import MySQLdb
import codecs
# import pandas as pd
import random
import argparse
import ConnectDB


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
	def __init__(self, selected_user, dbip, dbname, pwd):
		self.selected_user = selected_user
		conDB = ConnectDB.ConnectDB(dbip, dbname, pwd)
		self.cursor, self.db = conDB.connect_db()

	def graph_1month_select(self):
		print("Choosing the randomly selected users and time from 170w users......")
		with codecs.open("../graph_170w_1month.txt", mode='r', encoding='utf8') as graphfile:  # E:/data/social netowrks/weibodata
			# with codecs.open("graph_1month_selecteduser.txt", mode='a+', encoding='utf-8') as outfile:
			line = graphfile.readline()
			i = 0
			while line:
				'''if i < 3608200:
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
						(SELECT `:START_ID`, `type`, `time`, `:END_ID`, 
						FROM tb_miduserrelation WHERE `:START_ID` IN (%s) )""" % (in_p)
		self.cursor.execute(sql)
		self.db.commit()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-dbpwd",  help="Password of database")
	parser.add_argument("-dbIP", help="IP address of database")
	args = parser.parse_args()
	pwd = args.dbpwd
	dbip = args.dbIP

	SelectUser = SelectUser(select_user_num=10000, all_user_num=1787443)
	selected_user = SelectUser.random_select_users()
	with codecs.open('../selected_user.txt', mode='w', encoding='utf-8') as outfile:
		outfile.write(str(selected_user))
	BuildSubNetwork = BuildSubNetwork(selected_user, dbip=dbip, dbname='db_weibodata', pwd=pwd)
	BuildSubNetwork.graph_1month_select()
	BuildSubNetwork.user_relation_select()
	BuildSubNetwork.user_mid_select()