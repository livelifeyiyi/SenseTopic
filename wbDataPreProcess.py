import MySQLdb
import codecs
import pandas as pd
import random

class Data2DB:
	def __init__(self):
		pass

	def connect_db(self, dbname, pwd):
		db = MySQLdb.connect("192.168.2.134", "root", "%s" % pwd, '%s' % dbname, charset='utf8')
		cursor = db.cursor()
		sql = """use %s """ % dbname
		cursor.execute(sql)

	def graph_1month(self):

		pass

	def random_select_users(self, select_user_num, all_user_num):
		"""

		:param select_user_num: 
		:param all_user_num: 
		:return: 
		"""
		select_user =[]
		for i in range(select_user_num):
			select_user.append(random.randint(0, all_user_num))
		return select_user


if __name__ == '__main__':
	data2db = Data2DB()
	# data2db.connect_db('db_weibodata')