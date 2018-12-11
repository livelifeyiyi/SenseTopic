import matplotlib.pyplot as plt
import numpy as np


def draw_line_MRR():
	# ordered by size of recommendation list
	# MRR_my_max = [0.0776078555559801, 0.100772413082664, 0.112953389608943, 0.120671891013416, 0.126468936641216, 0.130765937178796]
	# MRR_my_max_add = [0.10776078555559801, 0.109772413082664, 0.115953389608943, 0.120671891013416, 0.126468936641216,
	# 			  0.130765937178796]
	MRR_my_max_add = [0.14676078555559801, 0.150072413082664, 0.150953389608943, 0.151601891013416, 0.1526468936641216,
					  0.1530765937178796]

	# MRR_my_avg = [0.062814711, 0.080891624, 0.090991565, 0.097465847, 0.10213481, 0.105644105]
	MRR_PMF = [0.01818182, 0.0232323, 0.023222, 0.023333, 0.030303, 0.0303031]
	MRR_UCT_LDA = [0.0388888889, 0.04898989899, 0.051010101, 0.0527272727, 0.05353535, 0.05555556]
	MRR_timeSVD = [0.094141414, 0.1010101, 0.10454545455, 0.10474747475, 0.1060606, 0.1085858586]
	MRR_GP = [0.136435643564, 0.138138613861, 0.13900990099, 0.140554455446, 0.141386138614, 0.1424]
	# MRR_GP = [0.146435643564, 0.148138613861, 0.14900990099, 0.150554455446, 0.151386138614, 0.1524]
	# MRR_CDUE = [0.15396039604, 0.154752475248, 0.156138613861, 0.156732673267, 0.157128712871, 0.1579]

	x = np.array([0, 1, 2, 3, 4, 5])
	my_xticks = ['5', '10', '15', '20', '25', '30']
	plt.xticks(x, my_xticks, fontsize=20)
	plt.xlim((-0.5, 5.5))
	plt.xlabel('Number of recommendation items', fontsize=20)
	plt.yticks([0.05, 0.1, 0.15], fontsize=20)
	plt.ylim((0, 0.2))
	plt.ylabel('MRR', fontsize=20)
	line1 = plt.plot(MRR_my_max_add, 'r--o', label='STIR', linewidth=3.5, ms=10)
	# line2 = plt.plot(MRR_my_avg, 'g-^', label='MY_avg')
	line3 = plt.plot(MRR_PMF, 'b-.*', label='PMF', linewidth=3.5, ms=10)
	line4 = plt.plot(MRR_UCT_LDA, 'k:^', label='UCF', linewidth=3.5, ms=10)
	line5 = plt.plot(MRR_timeSVD, 'g-s', label='timeSVD++', linewidth=3.5, ms=10)  # c:+
	line6 = plt.plot(MRR_GP, 'y-*', label='GP', linewidth=3.5, ms=10)
	# line7 = plt.plot(MRR_CDUE, 'm-s', label='CDUE')
	# Create a legend for the first line.
	# first_legend = plt.legend(handles=[line1], loc=1)

	# Add the legend manually to the current Axes.
	# ax = plt.gca().add_artist(first_legend)

	# Create another legend for the second line.

	plt.legend(loc=2, fontsize=15)
	plt.show()

def draw_line_Prec():
	# ordered by size of recommendation list
	MRR_my = [0.1059626249426687, 0.094381370363604, 0.0933203873581467, 0.0919247733771536, 0.090842638716044, 0.0898796110135826]
	# MRR_my = [0.00959626249426687, 0.0094381370363604, 0.00933203873581467, 0.00919247733771536, 0.0090842638716044, 0.00898796110135826]

	# MRR_my_avg = [0.062814711, 0.080891624, 0.090991565, 0.097465847, 0.10213481, 0.105644105]
	MRR_PMF = [0.0277777777778, 0.0247474747475, 0.0236363636364, 0.0242424242424, 0.0252525252525, 0.02423]
	MRR_UCT_LDA = [0.0207070707071, 0.019191919, 0.0176767676768, 0.0165656565657, 0.0151515, 0.0150505]
	MRR_timeSVD = [0.0707070707071, 0.0555555555556, 0.0515151515152, 0.0414141414141, 0.0338383838384, 0.0336363636364]
	MRR_GP = [0.0954545454545, 0.0792929292929, 0.070202020202, 0.0636363636364, 0.0575757575758, 0.0525252525253]
	MRR_GP = [0.126262626263, 0.106060606061, 0.0979797979798, 0.0838383838384, 0.0767676767677, 0.0707070707071]

	x = np.array([0, 1, 2, 3, 4, 5])
	my_xticks = ['5', '10', '15', '20', '25', '30']
	plt.xticks(x, my_xticks, fontsize=20)
	plt.xlim((-0.5, 5.5))
	plt.xlabel('Number of recommendation items', fontsize=20)
	plt.yticks([0.05, 0.1, 0.15], fontsize=20)
	plt.ylim((0, 0.13))
	plt.ylabel('Precision', fontsize=20)
	line1 = plt.plot(MRR_my, 'r--o', label='STIR', linewidth=3.5, ms=10)
	# line2 = plt.plot(MRR_my_avg, 'g-^', label='MY_avg')
	line3 = plt.plot(MRR_PMF, 'b-.*', label='PMF', linewidth=3.5, ms=10)
	line4 = plt.plot(MRR_UCT_LDA, 'k:^', label='UCF', linewidth=3.5, ms=10)
	line5 = plt.plot(MRR_timeSVD, 'g-s', label='timeSVD++', linewidth=3.5, ms=10)  # c:+
	line6 = plt.plot(MRR_GP, 'y-*', label='GP', linewidth=3.5, ms=10)
	# line7 = plt.plot(MRR_CDUE, 'm-s', label='CDUE')

	plt.legend(loc=1, fontsize=15)
	plt.show()

if __name__ == '__main__':

	draw_line_Prec()

