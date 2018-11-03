import codecs
import pandas as pd
import numpy as np


def format_vector(vector_file):
	"""
	split the words and vectors for visualization with tsne
	:param vector_file: trained vector file
	:return: 
	"""
	with codecs.open(vector_file, encoding='utf-8', mode='r') as inFile:
		with codecs.open('word.labels', encoding='utf-8', mode='a+') as labelFile:
			with codecs.open('word.vectors', encoding='utf-8', mode='a+') as vecFile:
				line = inFile.readline().strip('\n').strip('\r')
				while line:
					line_list = line.split(' ')
					word = line_list[0]
					vectors = line_list[1:]
					labelFile.write(word + '\n')
					vecFile.write(' '.join(vectors) + '\n')
					line = inFile.readline().strip('\n').strip('\r')


def get_topic_words(topic_word_file, vector_file):
	"""
	get the words in topic_word file to help with visualize
	:param topic_word_file: TopicWord file
	:return: topicWord.labels; topicWord.vectors
	"""
	vectors = pd.read_table(vector_file, sep='\s+', index_col=0, header=None)
	with codecs.open(topic_word_file, encoding='utf-8', mode='r') as inFile:
		with codecs.open('topicWord.labels', encoding='utf-8', mode='a+') as labelFile:
			with codecs.open('topicWord.vectors', encoding='utf-8', mode='a+') as vecFile:
				topic_words = []
				line = inFile.readline()
				i = 0
				while line:
					words = line.split(': ')[1]
					for each_word in words.split():
						if each_word not in topic_words:
							topic_words.append(each_word)
							labelFile.write(each_word + '\n')
							topic_word_vec = vectors.ix[each_word.encode('utf-8')].values
							vecFile.write(np.array2string(topic_word_vec, max_line_width=3000, formatter={'float_kind': lambda x: '%6f'%x})[1:-1] + '\n')
					line = inFile.readline()
					i += 1
					print i


wordVec_file = "E:\\code\\SN2\\SE-WRL-master\\Clean-Sogou\\vectors_split_sense.bin"
topic_word_file = "E:\\code\\SN2\\pDMM-master\\output\\model.filterAllstop.sense.100.topWords"
get_topic_words(topic_word_file, wordVec_file)