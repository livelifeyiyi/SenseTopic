import json
import codecs


class evaluatePrediction:
	def __init__(self, rootDir, user_num):
		self.rootDir = rootDir
		self.user_num = int(user_num)

	def evaluate(self):
		with codecs.open(self.rootDir + "Predict_Rij_t21.json", mode='r') as infile:
			Predict_Rij = json.load(infile)
			print len(Predict_Rij["20"])
			for user_id in range(self.user_num):
				print("********user_id********" + str(user_id))
				Rij = Predict_Rij["20"]["%s" % user_id]
				for each_id in range(len(Rij)):
					if Rij[each_id] != 0.0:
						print("index: " + str(each_id) + ", Rij: " + str(Rij[each_id]))

if __name__ == '__main__':
	evaluate = evaluatePrediction('./', 100)
	evaluate.evaluate()