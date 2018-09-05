# -*- coding: utf-8 -*-
import codecs
import json
# bound = [':', '}']
# r = open('HowNet.txt', 'r')
word_sememe_dict = {}

with codecs.open('HowNet_nonull.txt', encoding='utf-8', mode='r') as r:
	# def readItem():
	for i in range(209231):  # 212540  #209231
		# print i
		while r.readline()[:2] != 'NO':
			pass
		word = r.readline()[:-1].split('=')[1]
		while r.readline()[:3] != 'E_E':
			pass
		DEF = r.readline()[:-1]  # .split('=')[1]
		bound = [':', '}']
		maohao = []
		sememes = []
		sememe = ''
		begin = False
		for x in DEF:
			if x == '|':
				begin = True
				continue

			if x in bound:
				'''if x == ':':
					if x not in maohao:
						maohao.append(x)
					else:
						begin = False
						sememe = ''
						continue
					# bound = ['}']
					'''

				begin = False
				'''if sememe == word:
					sememe = ''
					continue'''
				if sememe != '' and (sememe not in sememes):
					sememes.append(sememe)
				sememe = ''
				continue
			if begin:
				sememe += x
		sememes = sorted(sememes)
		if word_sememe_dict.has_key(word):
			if sememes not in word_sememe_dict[word] and len(sememes) > 0 and sememes:
				word_sememe_dict[word].append(sememes)
		else:
			if sememes:
				word_sememe_dict[word] = [sememes]
		r.readline()
		r.readline()


# for i in range(212540):
# readItem()
# for key, value in word_sememe_dict.items():
# 	word_sememe_dict[key] = list(set(value))
# w = open('Word_Sense_Sememe.txt', 'w')

with codecs.open('Word_Sense_Sememe_5.txt', encoding='utf-8', mode='a+') as w:
	with codecs.open('Word_Sense_Sememe_word_5.txt', encoding='utf-8', mode='a+') as wf_word:
		i = 0
		for key, values in word_sememe_dict.items():
			wf_word.write(key + ' ' + str(i) + '\n')
			i += 1
			w.write(key)
			w.write(' ')
			w.write(str(len(values)))
			# w.write(' ')
			for value in values:
				# if len(values) == 1:  # and len(value) == 1 and value[0] == key:
				# 	continue
				w.write(' ')
				w.write(str(len(value)))
				# w.write(' ')
				for sememe in value:
					w.write(' ')
					w.write(sememe)
					
			w.write('\n')
# w.close()

