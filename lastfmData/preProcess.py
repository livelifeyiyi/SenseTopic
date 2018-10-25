import codecs
import pandas as pd
import numpy as np
import json


def get_distinct_userID():
	filename = rootDir + "user_friends.dat"
	with codecs.open(filename, mode='r', encoding='utf-8') as infile:
		infile.readline()
		user_friend = infile.readline().strip('\n').strip('\r')
		distinct_user = []
		i = 0
		while user_friend:
			print i
			i += 1
			user, friend = user_friend.split('\t')[0], user_friend.split('\t')[1]
			if user not in distinct_user:
				distinct_user.append(int(user))
			if friend not in distinct_user:
				distinct_user.append(int(friend))
			user_friend = infile.readline().strip('\n').strip('\r')
		distinct_user = sorted(distinct_user)
		print distinct_user


def get_index_tags():
	tag_id_info = pd.read_table(rootDir + 'tags.dat', sep='\t', header=0, index_col=0)

	tags = {}
	with codecs.open(rootDir + "tmp_item_tag.txt", mode='r', encoding='utf-8') as tagfile:
		tagfile.readline()
		info = tagfile.readline().strip('\n').strip('\r')
		while info:
			info_split = info.split('\t')
			# uID, aID, tag, year = info_split[0], info_split[1], info_split[2], info_split[5]
			aID, tag = int(info_split[0]), int(info_split[1])
			tag_value = tag_id_info.ix[tag].values[0]
			if not tags.has_key(aID):
				tags[aID] = [tag_value]
			else:
				tags[aID].append(tag_value)
			info = tagfile.readline().strip('\n').strip('\r')
	for key in tags.keys():
		print key
		with codecs.open(rootDir+'item_tags_doc', mode='a+', encoding='utf-8') as of:
				of.write(u' '.join(tags[key])+'\n')


def get_R_ui_t():
	user_tag_info = pd.read_table(rootDir + 'user_taggedartists.dat', sep='\t', header=0)
	user_num = 1892
	item_num = 12316  # 17632
	time_num = 8
	aID_id = np.loadtxt(rootDir+'artistsID_id_havetags_havevecs.txt')
	from selected_user import selected_user
	R_ua_t = np.zeros((time_num, user_num, item_num), dtype='int')
	with codecs.open(rootDir + "user_artists.dat", mode='r', encoding='utf-8') as uafile:
		uafile.readline()
		info = uafile.readline().strip('\n').strip('\r')
		i = 0
		while info:
			i += 1
			print i
			info_split = info.split('\t')
			uID, aID, weight = int(info_split[0]), int(info_split[1]), info_split[2]
			res = user_tag_info.loc[(user_tag_info['userID'] == uID) & (user_tag_info['artistID'] == aID)]
			if not res.empty:
				time = res.iloc[-1].values[-1]
				time_id = int(time)-2005+1
				if time_id <= 0:
					time_id = 0
			else:
				time_id = 0
			# new_line = info + str(time_id) + '\n'
			try:
				art_index = np.where(aID_id == aID)[0][0]
				user_index = selected_user.index(uID)
				R_ua_t[time_id][user_index][art_index] = weight
			except Exception as e:
				print e, aID
			info = uafile.readline().strip('\n').strip('\r')
	np.save(rootDir + 'Actual_Rij_t.npy', R_ua_t)


def get_neighbors():
	user_friends_pd = pd.read_table(rootDir+'user_friends.dat', header=0, sep='\t')
	# user_friends_pd = user_friends_pd.set_index(['userID'])
	from selected_user import selected_user   # user id_index map
	user_num = len(selected_user)
	time_num = 1
	print("Getting neighbors of each user at different time......")
	# flag = 0 return all neighbors, =1 return only friends.
	# ni_follow = dict.fromkeys([i for i in range(user_num)], '')
	follow_dict_flag0 = {}  # dict.fromkeys([i for i in range(0, time_num)], ni_follow)
	# ni_friend = dict.fromkeys([i for i in range(user_num)], list)
	# friend_dict_flag1 = dict.fromkeys([i for i in range(0, time_num)], ni_friend)
	for time in range(0, time_num):
		print("Time: " + str(time))
		follow_dict_flag0[time] = {}
		ni_follow = {}
		for user_id in range(user_num):
			print("User: " + str(user_id))
			user = selected_user[user_id]
			# df.loc[(df['column_name'] == some_value) & df['other_column'].isin(some_values)]
			# print userMidDF[(userMidDF[':START_ID'] == uidi) & (userMidDF['type'] == '1')][':END_ID']
			res_df = user_friends_pd.loc[(user_friends_pd['userID'] == user) | (user_friends_pd['friendID'] == user)]
			row_index = res_df.index
			# print row_num.values[0], type(row_num)
			# print row_index
			follow = []
			for i in row_index:
				res = res_df.ix[i].values
				user1, user2 = res[0], res[1]
				if user1 == user and user2 not in follow:
					# follow_dict_flag0[time][user_id].append(user2)
					follow.append(user2)
				if user2 == user and user1 not in follow:
					# follow_dict_flag0[time][user_id].append(user1)
					follow.append(user1)
				# user1, user2 = res[0], res[1]
			ni_follow[user_id] = str(follow)
		follow_dict_flag0[time] = ni_follow

		# neighbors = friends
	print("Writing files......")

	with codecs.open(rootDir+'neighbors_flag_0_1.json', mode='w') as fo:
		json.dump(follow_dict_flag0, fo)
	# with codecs.open(rootDir+'neighbors_flag_1.json', mode='w') as fo:
	# 	json.dump(friend_dict_flag1, fo)


def get_friend_type():
	# u1, u2, t
	user_friends_pd = pd.read_table(rootDir + 'user_friends.dat', header=0, sep='\t')
	# user_friends_pd = user_friends_pd.set_index(['userID'])
	from selected_user import selected_user  # user id_index map
	user_num = len(selected_user)
	time_num = 1
	print("Getting friend type of each pair of users at different time......")
	friend_type = np.zeros((time_num, user_num, user_num))
	for time in range(0, time_num):
		print("Time: " + str(time))
		for user1_id in range(user_num):
			print("User: " + str(user1_id))
			user1 = selected_user[user1_id]
			for user2_id in range(user1_id + 1, user_num):
				user2 = selected_user[user2_id]
				res_df = user_friends_pd.loc[
					((user_friends_pd['userID'] == user1) & (user_friends_pd['friendID'] == user2))
					| ((user_friends_pd['userID'] == user2) & (user_friends_pd['friendID'] == user1))]
				row_index = res_df.index
				if len(row_index) == 0:
					friend_type[time][user1_id][user2_id] = 0.0
					friend_type[time][user2_id][user1_id] = 0.0
				else:
					friend_type[time][user1_id][user2_id] = 1.0
					friend_type[time][user2_id][user1_id] = 1.0
	print("Saving to dictionary '../data/friend_type_uijt.npy'......")
	# print friend_type
	np.save(rootDir+'friend_type_uijt.npy', friend_type)


def get_doc_dictionary():
	from words_no_vecs import words_no_vecs
	dictionary = dict.fromkeys(words_no_vecs, 0)

	with codecs.open(rootDir+'item_tags_doc', mode='r') as infile:
		line = infile.readline().strip('\n')
		while line:
			words = line.split()
			newline = ''
			for word in words:
				if not dictionary.has_key(word):
					newline = newline + word + ' '
			with codecs.open(rootDir+'item_tags_doc_invec', mode='a+') as outfile:
				outfile.write(newline + '\n')
			line = infile.readline().strip('\n')


def get_google_vec():
	from gensim.models import KeyedVectors
	from doc_dictionary import doc_words
	model = KeyedVectors.load_word2vec_format('/home/xiaoya/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
	for word in doc_words:
		try:
			newline = word + ' ' + np.array2string(model.get_vector(word), max_line_width=10000)[1:-1]
		except Exception as e:
			print word
			continue
		with open('word_google_w2v', 'a+') as outfile:
			outfile.write(newline + '\n')


def filter_item_no_vec():
	artistsID_id_havetags = np.loadtxt(rootDir+'artistsID_id_havetags.txt', dtype='int')
	new_np = []
	tag = 0
	# with codecs.open(rootDir + 'item_tags_doc_invec', mode='r') as infile:
	infile = open(rootDir+'item_tags_doc_invec', 'r')
	for (num, line) in enumerate(infile):
		if line == '' or line is None or line == '\n':
			continue
			'''index = num - tag
			np.delete(artistsID_id_havetags, index, 0)
			tag += 1'''
		else:
			new_np.append(artistsID_id_havetags[num])
	for i in new_np:
		print i
	# np.savetxt(rootDir+'artistsID_id_havetags_havevecs.txt', np.asarray(new_np, dtype='int'))


if __name__ == '__main__':
	rootDir = "E:\\code\\SN2\\lastfm-2k\\"
	# get_distinct_userID()
	# get_index_tags()
	get_R_ui_t()
	# get_doc_dictionary()
	# get_google_vec()
	# filter_item_no_vec()
	# get_neighbors()
	# get_friend_type()
