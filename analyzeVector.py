# -*- coding: utf-8 -*-
import codecs
import numpy as np
import argparse
# import pandas as pd


def average_word_vector(vectorfile, outvectorfile):
	"""
	Calculate the average vector of all sense vectors for each word.
	:param vectorfile: input sememe based vector file
	:param outvectorfile: output the average vector file
	:return: 
	"""
	with codecs.open(vectorfile, encoding='utf-8', mode='r') as infile:
		with codecs.open(outvectorfile, encoding='utf-8', mode='a+') as outfile:
			for i, line in enumerate(infile):
				print i
				if i == 0:
					a, b, c = map(int, line.split()[:3])
					print('Number of sememes: {}\n'
						  'Number of words: {}\n'
						  'Dimension of vectors: {}'.format(a, b, c))
					outfile.write(str(b) + ' ' + str(c) + '\n')
				elif i > 462667:
					outfile.write(line)
				else:
					sline = line.split()
					word = sline[0]
					sense_num = int(sline[1])
					vectors = sline[2:sense_num*c+2]  # (sense_num*c+2)
					# extra = len(vectors) % sense_num
					vector_list = []
					for start in range(0, len(vectors), c):
						vector_list.append(list(map(float, vectors[start: start+c])))
					# print vector_list
					vector_array = np.array(vector_list)
					vector_mean = np.mean(vector_array, axis=0)
					# print np.array2string(vector_mean, max_line_width=200, formatter={'float_kind': lambda x: '%6f'%x})[1:-1].strip('\n')
					new_line = word + ' ' + np.array2string(vector_mean, max_line_width=2000, formatter={'float_kind': lambda x: '%6f'%x})[1:-1] + '\n'
					outfile.write(new_line)


class BuildSenseVec:
	def __init__(self, vecfile, vecmeanfile, vocabfile, docfile, contextnum, doc_outfile, vec_outfile, vec_outfile_bydoc):
		self.vec_file = vecfile
		self.vec_mean_file = vecmeanfile  # not used
		self.vocab_file = vocabfile
		self.doc_file = docfile
		self.word2IdVocabulary = {}		# word & id in vocabulary
		self.id2WordVocabulary = {}
		# self.word2IdDoc
		self.vectors = {}  # pd.DataFrame()
		self.vector_mean = {}
		#  pd.read_table(vecmeanfile, sep='\s+', index_col=0, names=[i for i in range(0, 200)], encoding='utf-8')

		self.vector_word_doc = {}  # pd.DataFrame()  # word vectors in document, different senses represent different words.
		self.context_num = contextnum
		self.doc_out_file = doc_outfile
		self.vec_out_file = vec_outfile
		self.vec_out_file_bydoc = vec_outfile_bydoc
		self.vocab_num = 0

	@property
	def build_doc_sense_vec(self):
		"""
		for each word, select a sense with max attention value.
		build the vector space with each sense represent a word.
		:return: self.vector_word_doc
		"""
		with codecs.open(self.vocab_file, encoding='utf-8', mode='r') as infile:
			line = infile.readline()
			i = 0
			while line:
				word = line.split()[0]
				if not self.word2IdVocabulary.has_key(word):
					# print i, word
					# else:
					self.word2IdVocabulary[word] = i
				if not self.id2WordVocabulary.has_key(i):
					self.id2WordVocabulary[i] = word
				line = infile.readline()
				i += 1
			self.vocab_num = len(self.word2IdVocabulary)
			print "vocabulary number:" + str(self.vocab_num)

		with codecs.open(self.vec_file, encoding='utf-8', mode='r') as vecfile:
			with codecs.open(self.vec_out_file, encoding='utf-8', mode='a+') as vec_outfile:

				for i, line in enumerate(vecfile):
					if i % 10000 == 0:
						print i
					# if i > 72:
					# 	break
					if i == 0:
						a, b, c = map(int, line.split()[:3])
						print('Number of sememes: {}\n'
							  'Number of words: {}\n'
							  'Dimension of vectors: {}'.format(a, b, c))
					elif i > 462667:
						sline = line.strip('\n').split()
						word = sline[0]
						vector_list = []
						vector_list.append(sline[1:])
						vector_array = np.array(vector_list)
						word_id = self.word2IdVocabulary[word]
						if not self.vectors.has_key(word_id):
							self.vectors[word_id] = vector_array
						# vector_mean = np.mean(vector_array, axis=0)
						if not self.vector_mean.has_key(word_id):
							self.vector_mean[word_id] = vector_array
						# vec_outfile.write(line)
					elif i > 462887:
						break
					else:
						sline = line.strip('\n').split()
						word = sline[0]
						sense_num = int(sline[1])
						vectors = sline[2:sense_num*c+2]  # (sense_num*c+2)
						vector_list = []
						for start in range(0, len(vectors), c):
							vector_list.append(list(map(float, vectors[start: start+c])))
						vector_array = np.array(vector_list)
						word_id = self.word2IdVocabulary[word]
						if not self.vectors.has_key(word_id):
							self.vectors[word_id] = vector_array
						vector_mean = np.mean(vector_array, axis=0)
						if not self.vector_mean.has_key(word_id):
							self.vector_mean[word_id] = vector_mean
						'''j = 0
						for each_sense_vec in vector_array:
							if len(vector_array) > 1:
								new_line = word + '_' + str(j) + ' ' + np.array2string(each_sense_vec, max_line_width=2000,
																	formatter={'float_kind': lambda x: '%6f' % x})[1:-1] + '\n'
								j += 1
							else:
								new_line = word + ' ' + np.array2string(each_sense_vec, max_line_width=2000,
																					   formatter={'float_kind': lambda
																						   x: '%6f' % x})[1:-1] + '\n'

							vec_outfile.write(new_line)'''

		with codecs.open(self.doc_file, encoding='utf-8', mode='r') as docfile:
			with codecs.open(self.doc_out_file, encoding='utf-8', mode='a+') as doc_outfile:
				with codecs.open(self.vec_out_file_bydoc, encoding='utf-8', mode='a+') as vec_outfile_bydoc:
					print "Processing document file......"
					line = docfile.readline().strip('\n')
					while line:
						words = line.split()
						new_words = [x for x in words]
						for i in range(len(words)):
							word_id = self.word2IdVocabulary[words[i]]
							sense_vecs = self.vectors[word_id]
							sense_num = len(sense_vecs)
							if sense_num > 1:
								context_words = []
								for x in range(i-int(self.context_num), i+int(self.context_num)+1):
									if x != i and 0 <= x < len(words):
										context_words.append(words[x])
								sense_index = self.select_attention(context_words, sense_vecs)
								word_vec_i = sense_vecs[sense_index]
								new_wordi = words[i] + '_' + str(sense_index)
								self.vector_word_doc[new_wordi.encode('utf-8')] = word_vec_i
								new_words[i] = new_wordi

							else:
								word_vec_i = sense_vecs[0]
								self.vector_word_doc[words[i].encode('utf-8')] = word_vec_i
							vec_outfile_bydoc.write(new_words[i] + ' ' + np.array2string(word_vec_i, max_line_width=2000,
																	 formatter={'float_kind': lambda x: '%6f' % x})[1:-1] + '\n')

						doc_outfile.write(' '.join(new_words) + '\n')

						line = docfile.readline()

		return self.vector_word_doc

	def select_attention(self, context_words, sense_vecs):
		context_mean = np.zeros((1, 200))
		sense_index = 0
		for context in context_words:
			# context_mean += np.array(self.vector_mean[context.encode('utf-8')])
			try:
				word_id = self.word2IdVocabulary[context]
				context_mean += np.array(self.vector_mean[word_id], dtype=float)
			except Exception as e:
				print e
				print context, word_id
				# print self.vector_mean[word_id]
		context_mean /= float(len(context_words))
		sum_exp = 0.0
		for sense_vec in sense_vecs:
			sum_exp += np.exp(np.dot(np.array(context_mean, dtype=float), np.array(sense_vec, dtype=float)))

		att_sense = []
		for sense_vec in sense_vecs:
			att_sense.append(np.exp(np.dot(np.array(context_mean, dtype=float), np.array(sense_vec, dtype=float))) / sum_exp)
		try:
			sense_index = att_sense.index(max(att_sense))
		except Exception as e:
			print e
			print att_sense
			# att_sense = np.dot(context_mean, sense_vec) / sum_exp
			# sum_att += np.dot(sense_vec, att_sense)
		# return sum_att / float(len(sense_vecs))
		return sense_index


if __name__ == '__main__':
	'''vectorfile = 'SE-WRL/vectors.bin'
	outvectorfile = 'vector_mean_v2.bin'
	average_word_vector(vectorfile, outvectorfile)'''
	parser = argparse.ArgumentParser()
	parser.add_argument("-vec", default="../data/vectors_update.bin", help="The sense based word2vec file")
	parser.add_argument("-vecmean", default="../data/vector_mean_update.bin", help="The mean word2vec file")
	parser.add_argument("-vocab", default="../data/VocabFile", help="The vocabulary file")
	parser.add_argument("-doc", default="../data/root_content_noid_seged_filtstop_filtvec", help="The segmented document/corpus file")
	parser.add_argument("-context", default="2", help="The number of context words")
	parser.add_argument("-doc_out", default="../data/root_content_noid_seged_filtstop_wordsense",
						help="The output file with each word represented with the sense word")
	parser.add_argument("-vec_out", default="null",
						help="The output vector file with each word splited into sense word vector")
	parser.add_argument("-vec_out_doc", default="../data/vectors_split_sense_bydoc_notcutall.bin",
						help="The output vector file with each word splited into sense word vector only contain the words occured in document")

	args = parser.parse_args()
	vecfile = args.vec
	vecmeanfile = args.vecmean
	vocabfile = args.vocab
	docfile = args.doc
	# 上下文词个数
	contextnum = args.context
	# 将单词按不同sense重新组合文档
	doc_outfile = args.doc_out
	# 词向量拆分为每个sense对应一个词向量
	vec_outfile = args.vec_out
	# 词向量拆分为每个sense对应一个词向量，仅包含文档中出现的词
	vec_outfile_bydoc = args.vec_out_doc
	SenseVec = BuildSenseVec(vecfile, vecmeanfile, vocabfile, docfile, contextnum, doc_outfile, vec_outfile,
							 vec_outfile_bydoc)
	SenseVec.build_doc_sense_vec()

	'''rootdir = '../'
	vecfile = rootdir + 'data/vectors_update.bin'
	vecmeanfile = rootdir + 'data/vector_mean_update.bin'
	vocabfile = rootdir + 'data/VocabFile'
	docfile = rootdir + 'data/root_content_noid_seged_filtstop_filtvec'  # root_content_noid_seged_cutall_filter
	# 上下文词个数
	contextnum = 2
	# 将单词按不同sense重新组合文档
	doc_outfile = rootdir + 'data/root_content_noid_seged_filtstop_wordsense'
	# 词向量拆分为每个sense对应一个词向量
	vec_outfile = rootdir + 'data/vectors_split_sense.bin'
	# 词向量拆分为每个sense对应一个词向量，仅包含文档中出现的词
	vec_outfile_bydoc = rootdir + 'data/vectors_split_sense_bydoc_notcutall.bin'' '''

	'''att_sense = [8,8,2,8]
	sense_index = att_sense.index(max(att_sense))
	print sense_index'''