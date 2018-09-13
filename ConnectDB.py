import MySQLdb
class ConnectDB:
	def __init__(self, dbip, dbname, pwd):
		self.dbname = dbname
		self.pwd = pwd
		self.dbip = dbip

	def connect_db(self):
		db = MySQLdb.connect("%s" % self.dbip, "root", "%s" % self.pwd, '%s' % self.dbname, charset='utf8')
		cursor = db.cursor()
		# sql = """use %s """ % self.dbname
		# cursor.execute(sql)
		return cursor, db