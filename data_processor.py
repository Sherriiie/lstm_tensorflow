#!/usr/bin python3.4
# -*- coding: UTF-8 -*-

import numpy as np
import random
import os


# def __init__:


def test():
	print("test\n")


def buildVocab(file_path):
	vocab = {}
	code = 0
	vocab['UNKNOWN'] = int(code)
	for line in open(file_path + 'data/train'):
		items = line.strip().split(' ')
		for i in range(2, 4):
			words = items[i].split('_')
			for word in words:
				if not word in vocab:
					code += 1
					vocab[word] = code

	for line in open(file_path + 'data/val'):
		items = line.strip().split(' ')
		for i in range(2, 4):
			words = items[i].split('_')
			for word in words:
				if not word in vocab:
					code += 1
					vocab[word] = code
	print('The number of words in the dictionary is {}\n'.format(code))
	return vocab


def encode(slist, vocab):
	# print('--- The length of slist is {} ---'.format(len(slist)))
	slist_encoded = []
	for i in range(0, len(slist)):
		if slist[i] in vocab:
			slist_encoded.append(vocab[slist[i]])
		else:
			slist_encoded.append(vocab['UNKNOWN'])
	# print(slist_encoded)
	return slist_encoded


def loadDataSets(vocab, file_path, sequence_length, flag):
	myFile = ''
	if flag == 'train':
		myFile = file_path + 'data/train'
	elif flag == 'val':
		myFile = file_path + 'data/val'
	else:
		print ('--- Flag of datasets is wrong! ---')
	dataSets = []  # contains dataSets[[question,answer],[],[],...]
	if not os.path.exists(myFile):
		print('--- file doesnot exist ---\n')
	for line in open(myFile):
		items = line.strip().split(' ')
		items[2] = items[2].split('_')[0:(sequence_length)]
		items[3] = items[3].split('_')[0:(sequence_length)]
		items[2] = encode(items[2], vocab)
		items[3] = encode(items[3], vocab)
		dataSets.append([items[2], items[3]])  # [question,answer]
	# print('--- length of item[3] is {}, items[3] ---{}'.format(len(items[3]), items[3]))
	# print('length of dataSets is {}, shape is {}\n'.format(len(dataSets), np.shape(dataSets)))	#(size, 2, 200)
	return dataSets

#
# def loadTrainData(vocab, file_path, sequence_length, size):
# 	input_x1 = []  # questions
# 	input_x2 = []  # positive answers
# 	input_x3 = []  # negative answers
# 	trainDataSets = loadDataSets(vocab, file_path, sequence_length, 'train')
# 	# print('--- train dataSets = {}---\n'.format(trainDataSets))
# 	# print("shape of trainDataSets  is {}".format(np.shape(trainDataSets)), type(trainDataSets))
# 	trainData = random.sample(trainDataSets, size)  # choose [size] sets of train data randomly
#
# 	for i in range(size):
# 		# print('--- trainData = {} ---\n'.format(trainData))
# 		input_x1.append(trainData[i][0])
# 		input_x2.append(trainData[i][1])
# 		random_sample = random.sample(trainDataSets, 1)  # include question and answer
# 		# print(random_sample)		# shape is (1, 2, 200)
# 		# while (trainData[i][0] == random_sample[0][0]):
# 		# 	random_sample = random.sample(trainDataSets, 1)  # include question and answer
# 		input_x3.append(random_sample[0][1])
# 	return np.array(input_x1), np.array(input_x2), np.array(input_x3)


def loadTrainData(vocab, file_path, sequence_length, size, step):
    input_x1 = []       # questions
    input_x2 = []	    # positive answers
    input_x3 = []	    # negative answers
    trainDataSets = loadDataSets(vocab, file_path, sequence_length, 'train')
    # print('--- train dataSets = {}---\n'.format(trainDataSets))
    # print("shape of trainDataSets  is {}".format(np.shape(trainDataSets)),type(trainDataSets))
    trainData = list()
    num_data_sets = len(trainDataSets)
    for i in range(size):
	    trainData.append(trainDataSets[(step*size+i)%num_data_sets])

    for i in range(size):
        # print('--- trainData = {} ---\n'.format(trainData))
        input_x1.append(trainData[i][0])
        input_x2.append(trainData[i][1])
        random_sample = random.sample(trainDataSets,1)	# include question and answer
        while (trainData[i][0] == random_sample[0][0]):
	        random_sample = random.sample(trainDataSets, 1)  # include question and answer
	        print ('Re-random~~~')
        #print(random_sample)		# shape is (1, 2, 200)
        input_x3.append(random_sample[0][1])
    return np.array(input_x1), np.array(input_x2), np.array(input_x3)


def loadValDataSets(vocab, file_path, sequence_length):
	valDataSets = loadDataSets(vocab, file_path, sequence_length, 'val')
	return valDataSets


def loadValData(vocab, file_path, sequence_length, ratio):
	# print ('## load val data ##')
	input_x1 = []  # questions
	input_x2 = []  # positive answers
	input_x3 = []  # negative answers
	valDataSets = loadDataSets(vocab, file_path, sequence_length, 'val')
	random_sample = random.sample(valDataSets, 1)
	# print('/n------------random sample ',random_sample)
	for i in range(ratio):
		input_x1.append(random_sample[0][0])
		input_x2.append(random_sample[0][1])
		random_ans = random.sample(valDataSets, 1)  # include question and answer
		while (random_sample[0][0] == random_ans[0][0]):
		 	random_ans = random.sample(valDataSets, 1)  # include question and answer
			print ('=======Re-random~~~')
		input_x3.append(random_ans[0][1])
	# print(random_sample)
	return np.array(input_x1), np.array(input_x2), np.array(input_x3)


def loadValData_v2(vocab, file_path, sequence_length, ratio):       # load a batch of data once
	input_x1 = []  # questions
	input_x2 = []  # positive answers
	input_x3 = []  # negative answers
	valDataSets = loadDataSets(vocab, file_path, sequence_length, 'val')
	random_sample = random.sample(valDataSets, 1)
	input_x1.append(random_sample[0][0])        # a question with one postive answer and (ratio) negtive answers
	input_x2.append(random_sample[0][1])
	# print('/n------------random sample ',random_sample)
	for i in range(ratio):
		random_ans = random.sample(valDataSets, 1)  # include question and answer
		input_x3.append(random_ans[0][1])
	# print(random_sample)
	return np.array(input_x1), np.array(input_x2), np.array(input_x3)


def getSentence(data, vocab):
	de_vocab = deVocab(vocab)
	sentences = []
	for i in range(len(data)):
		temStrList = []
		for j in data[i]:
			tem = de_vocab[j]
			if not tem == '<a>':
				temStrList.append(tem)
		sentence = " ".join(temStrList)
		sentences.append(sentence)
	return sentences


def deVocab(vocab):
	de_vocab = {}
	for i in vocab:
		if not vocab[i] in de_vocab:
			de_vocab[vocab[i]] = i
	return de_vocab


def saveData(sentence):
	f = open("./data/saved_test_data.txt", 'a')
	f.write(sentence)
	f.flush()
	f.close()


def saveFeatures(cos12, cos13, loss , accuracy):
	f = open("./data/saved_features.txt", 'a')
	f.write('\nloss = ' + str(loss) + ', ')
	f.write('accuracy = ' + str(accuracy) + '\n')
	f.write('cos12 = ')
	temStr = ''
	for i in cos12:
		temStr += str(i) + ' '
	f.write(temStr + '\n')
	f.write('cos13 = ')
	temStr = ''
	for i in cos13:
		temStr += str(i) + ' '
	f.write(temStr + '\n')
	f.flush()
	f.close()


if __name__ == '__main__':
	filePath = './'
	vocab = buildVocab(filePath)
	q, ap, an = loadValData(vocab, filePath, 200, 1000)

	question = getSentence(q, vocab)
	ans_pos = getSentence(ap, vocab)
	ans_neg = getSentence(an, vocab)
	print (np.shape(question))
	# print (question)
	# print (ans_pos)
	# print (ans_neg)
	print ('Question: ' + question[0])
	print ('Ans_pos: ' + ans_pos[0])
	print ('Ans_neg: ' + ans_neg[0])


	# # 显示/保存测试数据
	'''
	sen_y1 = getSentence(input_y1,vocab)[0]
	sen_y2 = getSentence(input_y2,vocab)[0]
	sen_y3 = getSentence(input_y3,vocab)
	saveData('\nQuestion ' + ':\n'+ sen_y1)
	saveData('\nPositive Answer:\n'+ sen_y2)
	saveData('\nNegative Answers:\n')
	for i in sen_y3:
	saveData(i)'''

	# 保存cos12, cos13
	# saveFeatures([1, 3, 5, 78, 567, 836], [34, 56, 47, 444, 8998, 45], 0.1)
