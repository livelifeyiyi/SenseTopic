import numpy as np


class ProfileEvolution:
	def __init__(self):
		pass

	def cost_function(self):
		pass

	def minimum_Uit(self, user, item_set, time, gamma, eta, lambda_U):
		# user i,  time t
		user_h = []  # ..
		gamma_i = gamma[user]

		sum_userh = 0.0
		for h in user_h:
			gamma_h = gamma[h]
			eta_h = eta[h]
			sum_userh += gamma_h * self.L_hit(h, user, time, eta_h) * (self.Uit_hat(h, time + 1) - self.U_it(h, time + 1))
		min_Uit = 0.0
		for item in item_set:
			# item j
			min_Uit += self.Y_ijt(user, item, time) * (np.dot(U_it, self.V_j(item)) - self.R_ijt(user, item, time)) * self.V_j(item)
			+ lambda_U*(self.U_it(user, time)-self.Uit_hat(user, time, gamma_i))
			+ lambda_U*(1-gamma_i)(self.U_it(user, time+1)-self.Uit_hat(user, time+1, gamma_i))
			+ lambda_U*sum_userh

	def minimum_gamma(self, user, lambda_U, time_sequence):
		# user i
		gamma_i = 0  # sample gamma to minimum target function

		sum_t = 0.0
		for t in time_sequence:
			neighbors_i = self.neighbors(user, t - 1, 0)
			sum_h = 0.0
			for h in neighbors_i:
				sum_h += self.L_hit(h, user, t - 1) * self.U_it(h, t - 1)
			sum_t += (self.Uit_hat(user, t, gamma_i) - self.U_it(user, t)) * (sum_h-self.U_it(user, t - 1))
		min_target = lambda_U * sum_t

	def minimum_eta(self, user_i, lambda_U, time_sequence, gamma):
		sum_t = 0.0
		gamma_i = gamma[user_i]
		for t in time_sequence:
			neighbors_i = self.neighbors(user_i, t - 1, 0)
			sum_h = 0.0
			for user_h in neighbors_i:

				is_friend = 0  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends
				friends_i = self.neighbors(user_i, t, 1)
				friends_h = self.neighbors(user_h, t, 1)
				intersec = list(set(friends_i).intersection(set(friends_h)))
				sum_h += (is_friend - float(len(intersec) / len(friends_i))) * self.U_it(user_h, t-1)

			sum_t += (self.Uit_hat(user_i, t, gamma_i) - self.U_it(user_i, t)) * (gamma_i * sum_h + (1-gamma_i) * self.U_it(user_, t-1))
		min_target = lambda_U * sum_t

	def Y_ijt(self, user_i, item_j, time):
		# Y_ijt = 1 if user_i has a link with item_j at time t, else=0
		pass

	def R_ijt(self, user_i, item_j, time):
		# R_ijt = rating preference score of user_i to item_j at time t
		pass

	def V_j(self, item):
		# V_j = topic of item_j
		pass

	def U_it(self, user, time):
		# U_it = ......
		# U_it
		pass

	def Uit_hat(self, user, time, gamma_i):  # user i
		neighbors_i = self.neighbors(user, time-1, 0)
		sum_h = 0.0
		for h in neighbors_i:
			sum_h += self.L_hit(h, user, time-1) * self.U_it(h, time-1)
		Uit_hat = (1-gamma_i) * self.U_it(user, time-1) + gamma_i * sum_h
		return Uit_hat

	def neighbors(self, user, time, flag):
		# flag = 0 return all neighbors, =1 return only friends.
		neighbors = []  # list of the users who have a link with user_i
		return neighbors  # save into a global parameter

	def L_hit(self, user_h, user_i, time, eta):
		is_friend = 0  # =0 if user_h has no link with user_i, =0.5 if they are one way fallow, =1 if they are friends
		friends_i = self.neighbors(user_i, time, 1)
		friends_h = self.neighbors(user_h, time, 1)
		intersec = list(set(friends_i).intersection(set(friends_h)))
		L_hit = eta * is_friend + (1-eta) * float(len(intersec)/len(friends_i))
		return L_hit


if __name__ == '__main__':
	Profile = ProfileEvolution()
	