# -*- coding: utf-8 -*-
import argparse

import jieba
import codecs
import pandas as pd
import linecache
from stop_word import stopwords
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# dataset_dir = 'E:\\data\\social netowrks\\weibodata\\processed\\'


def add_vocab_file():
	word2IdVocabulary = {}
	with codecs.open('data/extra_vocab', encoding='utf-8', mode='r') as infile:
		with codecs.open('data/VocabFile', encoding='utf-8', mode='a+') as outfile:
			line1 = outfile.readline()
			i = 0
			while line1:
				word = line1.split()[0]
				if not word2IdVocabulary.has_key(word):
					# print i, word
					# else:
					word2IdVocabulary[word] = i
				line1 = outfile.readline()
				i += 1

			line = infile.readline()
			while line:
				word_new = line.split()[0]
				if not word2IdVocabulary.has_key(word_new):
					outfile.write(word_new + '\n')
				line = infile.readline()


def target_to_462667():
	need_to_add_vocab = []
	with codecs.open('data/need_to_add_vocab', encoding='utf-8', mode='r') as addfile:
		for i, line in enumerate(addfile):
			word = line.strip('\n').strip('\r')
			need_to_add_vocab.append(word)
	with codecs.open('data/vectors.bin', encoding='utf-8', mode='r') as infile:
		with codecs.open('data/vectors_update.bin', encoding='utf-8', mode='a+') as outfile:
			for i, line in enumerate(infile):
				if i < 462668:
					outfile.write(line)
				else:
					word = line.split()[0]
					if word in need_to_add_vocab:
						print word
						outfile.write(line)
			print "the end"


def seg_with_dictionary():
	"""
	word segmentation with user dictionary
	:return: 
	"""
	jieba.load_userdict('data/VocabFile')
	contentFile = 'data/root_content_noid_utf8'
	segFile = 'data/root_content_noid_seged_cutall'
	with codecs.open(contentFile, encoding='utf-8', mode='r') as infile:
		doc = infile.read()
		seg_list = jieba.cut(doc, cut_all=True)
		str_default = " ".join(seg_list)
		with codecs.open(segFile, encoding='utf-8', mode='a+') as outfile:
			outfile.write(str_default)


def filter_stopwords():
	segFile = 'data/root_content_noid_seged'
	resFile = 'data/root_content_noid_seged_filtstop'
	with codecs.open(segFile, encoding='utf-8', mode='r') as infile:
		text = infile.readline()
		while text:
			restext = ''
			words = text.split(' ')
			for word in words:
				if word not in stopwords:
					restext += word
					restext += ' '
			with codecs.open(resFile, encoding='utf-8', mode='a+') as outfile:
				outfile.write(restext)
			text = infile.readline()


def filter_stop_not_in_vector(segFile, resFile):
	"""
	filter stop words and wods not in word vectors 
	:return: 
	"""
	# read vocab file
	vector_vocab = pd.read_table('../data/VocabFile', index_col=0, sep='\s+', names=['frequency'])

	with codecs.open(segFile, encoding='utf-8', mode='r') as infile:
		#for i, text in enumerate(infile):
		i = 0
		text = infile.readline().strip('\n')
		while text:
			print i
			restext = ''
			words = text.split(' ')
			for word in words:
				# print word
				if word not in stopwords:
					if word.encode('utf-8') in vector_vocab.index:
						restext += word
						restext += ' '
			with codecs.open(resFile, encoding='utf-8', mode='a+') as outfile:
				outfile.write(restext + '\n')
				text = infile.readline().strip('\n')
				i += 1


def check_file_encode():
	from chardet.universaldetector import UniversalDetector
	bigdata = open('SE-WRL/corpus/Clean-Sogou-all.txt', 'rb')
	detector = UniversalDetector()
	for line in bigdata.readlines():
		detector.feed(line)
		if detector.done:
			break
	detector.close()
	bigdata.close()
	print(detector.result)


def cal_frequency(filename, outfile):
	"""
	calculate word frequency of the input file 
	:return: 
	"""
	word_dict = {}
	with codecs.open(filename, encoding='utf-8', mode='r') as wf:
		line = wf.readline()
		while line:
			LineWords = line.strip('\n').strip('\r').split(' ')
			for word in LineWords:
				if word in word_dict.keys():
					word_dict[word] += 1
				else:
					word_dict[word] = 1

	print len(word_dict)
	with codecs.open(outfile, encoding='utf-8', mode='a+') as of:
		for key in word_dict.keys():
			of.write(key + ' ' + word_dict[key] + '\n')


def replace_sememe_file(dataset_sememe, hownet_sememe, out_file):
	hownet_word = pd.read_table('Word_Sense_Sememe_word_5.txt', encoding='utf-8', sep='\s+', index_col=0, names=['linenum'])
	with codecs.open(dataset_sememe, encoding='utf-8', mode='r') as fd:
		# with codecs.open(hownet_sememe, encoding='utf-8', mode='r') as fh:
		with codecs.open(out_file, encoding='utf-8', mode='a+') as outfile:
			line = fd.readline()
			while line:
				word = line.split(' ')[0]
				if word in hownet_word.index:  # .values
					print word
					linenum = hownet_word.ix[word, ['linenum']].values
					hownet_line = linecache.getline(hownet_sememe, int(linenum))
					outfile.write(hownet_line)
				else:
					outfile.write(line)
				line = fd.readline()


def fill_one_sememe(infile, outfile):
	with codecs.open(infile, encoding='utf-8', mode='r') as inf:
		with codecs.open(outfile, encoding='utf-8', mode='a+') as outf:
			line = inf.readline().strip('\n')
			while line:
				words = line.split(' ')
				if len(words) == 2 and words[1] == '1':
					newline = line + ' ' + words[1] + ' ' + words[0]
					outf.write(newline + '\n')
				else:
					outf.write(line + '\n')
				line = inf.readline().strip('\n')


if __name__ == '__main__':
	# cal_frequency(dataset_dir+'root_content_noid_seg.txt', 'word_frequency.txt')
	# hownet_sememe = 'Word_Sense_Sememe_5.txt'
	# dataset_sememe = 'Word_Sense_Sememe_File'
	# print linecache.getline(hownet_sememe, 1)

	# replace_sememe_file(dataset_sememe, hownet_sememe, 'Word_Sense_Sememe_Replaced_5')
	# fill_one_sememe('Word_Sense_Sememe_Replaced_5', 'Word_Sense_Sememe_Replaced_5.2')
	# seg_with_dictionary()
	# filter_stopwords()
	# add_vocab_file()
	# target_to_462667()

	# segFile = '../data/root_content_noid_seged_filtstop'
	# resFile = '../data/root_content_noid_seged_filtstop_filtvec'
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--segFile", default="../data/root_content_noid_seged_filtstop", help="The seged file")
	parser.add_argument("-r", "--resFile", default="../data/root_content_noid_seged_filtstop_filtvec", help="The file filtered words not in vectors")
	args = parser.parse_args()
	segFile = args.s
	resFile = args.r
	filter_stop_not_in_vector(segFile, resFile)



