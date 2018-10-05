import codecs


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

format_vector("E:\\code\\SN2\\SE-WRL-master\\Clean-Sogou\\vectors_split_sense.bin")