# -*- coding: utf-8 -*-

#------------------------------First import all the necessary packeges-------------------------------------

from collections import Counter
from collections import Iterable
import operator
from Ngram_Numeric import NumConver
import unicodedata
from nltk.book import *
from nltk import sent_tokenize, word_tokenize, pos_tag
import csv
from nltk.corpus import stopwords
import itertools
import nltk
import numpy as np
import sys
import csv
import time
import math
start_time = time.time()
reload(sys)
sys.setdefaultencoding("utf-8")
global stopwords
global delimiters
#There are several words that are not specifically important. These words are used in any corpus, and
#do nat have special meaning for any particular corpus. Let us declare those as stopwords.

stopwords = ["a","able","about","above","according","accordingly","across","actually","after",
"afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already",
"also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow",
"anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are",
"aren't","around","as","a's","aside","ask","asking","associated","at","available","away","awfully",
"be","became","because","become","becomes","becoming","been","before","beforehand","behind","being",
"believe","below","beside","besides","best","better","between","beyond","both","brief","but","by",
"came","can","cannot","cant","can't","cause","causes","certain","certainly","changes","clearly",
"c'mon","co","com","come","comes","concerning","consequently","consider","considering","contain",
"containing","contains","corresponding","could","couldn't","course","c's","currently","dear",
"definitely","described","despite","did","didn't","different","do","does","doesn't","doing",
"done","don't","down","downwards","during","each","edu","eg","eight","either","else","elsewhere",
"enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything",
"everywhere","ex","exactly","example","except","far","few","fifth","first","five","followed","following",
"follows","for","former","formerly","forth","four","from","further","furthermore","get","gets","getting",
"given","gives","go","goes","going","gone","got","gotten","greetings","had","hadn't","happens","hardly",
"has","hasn't","have","haven't","having","he","hello","help","hence","her","here","hereafter","hereby",
"herein","here's","hereupon","hers","herself","he's","hi","him","himself","his","hither","hopefully",
"how","howbeit","however","i","i'd","ie","if","ignored","i'll","i'm","immediate","in","inasmuch","inc",
"indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't",
"it","it'd","it'll","its","it's","itself","i've","just","keep","keeps","kept","know","known","knows",
"last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely",
"little","look","looking","looks","ltd","mainly","many","may","maybe","me","mean","meanwhile","merely",
"might","more","moreover","most","mostly","much","must","my","myself","name","namely","nd","near","nearly",
"necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none",
"noone","nor","normally","not","nothing","novel","now","nowhere","obviously","of","off","often","oh","ok",
"okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours",
"ourselves","out","outside","over","overall","own","particular","particularly","per","perhaps","placed",
"please","plus","possible","presumably","probably","provides","que","quite","qv","rather","rd","re","really",
"reasonably","regarding","regardless","regards","relatively","respectively","right","said","same","saw","say",
"saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves",
"sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six",
"so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon",
"sorry","specified","specify","specifying","still","sub","such","sup","sure","take","taken","tell","tends",
"th","than","thank","thanks","thanx","that","thats","that's","the","their","theirs","them","themselves",
"then","thence","there","thereafter","thereby","therefore","therein","theres","there's","thereupon","these",
"they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though",
"three","through","throughout","thru","thus","tis","to","together","too","took","toward","towards","stopwordsd",
"stopwordss","truly","try","trying","t's","twas","twice","two","un","under","unfortunately","unless","unlikely",
"until","unto","up","upon","us","use","used","useful","uses","using","usually","value","various","very","via","viz",
"vs","want","wants","was","wasn't","way","we","we'd","welcome","well","we'll","went","were","we're","weren't",
"we've","what","whatever","what's","when","whence","whenever","where","whereafter","whereas","whereby","wherein",
"where's","whereupon","wherever","whether","which","while","whither","who","whoever","whole","whom","who's","whose",
"why","will","willing","wish","with","within","without","wonder","won't","would","wouldn't","yes","yet","you","you'd","you'll","your","you're","yours","yourself","yourselves","you've","zero"]

textpath =  'input.csv'
delimiters=[".",",",";",":","?","/","!","'s","'ll","'d","'nt"]

sentenceugs = []
sentencebgs = []
sentencetgs = []
counts2 = []
counts3 = []

#The following class "Unigram" is used to calculate unigrams
class Unigram:
	def __init__(self,str):
		self.str = str
	def compute_unigrams(self,str):
		ugs = [word for word in str if (word not in stopwords and word not in delimiters)]
		return ugs

#The following class "Bigram" is used to calculate bigrams

class Bigram:
	def __init__(self,str):
		self.str = str
	def compute_bigrams(self,str):
		final_list=[]
		bigramtext = list(nltk.bigrams(str))
		for item in bigramtext:
			if item[0] not in delimiters and item[len(item)-1] not in delimiters:
				if not item[0].isdigit() and not item[1].isdigit():
					if item[0] not in stopwords and item[len(item)-1] not in stopwords:
						if len(item[0])>1  and len(item[len(item)-1])>1:
							final_list.append(item)
		return final_list

#The following class "Trigrams" is used to calculate trigrams

class Trigram:
	def __init__(self,str):
		self.str = str
	def compute_trigrams(self,str):
		final_list=[]
		trigramtext = list(nltk.trigrams(str))
		for item in trigramtext:
			if item[0] not in delimiters and item[1] not in delimiters and item[len(item)-1] not in delimiters:
				if not item[0].isdigit() and not item[1].isdigit() and not item[len(item)-1].isdigit():
					if item[0] not in stopwords and item[len(item)-1] not in stopwords:
						if len(item[0])>1  and len(item[len(item)-1])>1:
							final_list.append(item)
		return final_list
Sentset = set()

class Sentence:
	#----------------Global variable declaration------------
	global sents
	global tokens
	global counts1
	global counts2
	global counts3
	global number_of_text
	text=[]
	counts1=[]

	l1=[]
	l2=[]
	l3=[]

#----------------Here the input is taken from the following file----------------


	data = csv.reader(open(textpath, 'r'), delimiter=",", quotechar='|')
	text= []
	rowt= []
	itteration_count=0
	s =""
	for row in data:
		 #print len(row[2])
		if itteration_count>0:
			row[0] = unicode(row[0], errors='ignore')
			text.append(row[0])
		itteration_count=itteration_count+1
	for text_index in text:
			text_index = text_index.lower()
	   #-----------------Sentence representation-----------------
			sents = sent_tokenize(text_index)

			sentences = [nltk.word_tokenize(sent) for sent in sents]
			#------------------Word representation------------------------
			tokens = [nltk.word_tokenize(sent) for sent in sents]

	   #---------Object Creation of Unigram,Bigram and Trigram class--------
			unigram_obj = Unigram("")
			bigram_obj = Bigram("")
			trigram_obj = Trigram("")

			sentenceugs = []
			sentencebgs = []
			sentencetgs = []

			#----------Accessing the computed List of Unigram,Bigram and Trigram---------
			for token in tokens:

	  			a = unigram_obj.compute_unigrams(token)
				sentenceugs.append(a)
				b = bigram_obj.compute_bigrams(token)
				sentencebgs.append(b)
				c = trigram_obj.compute_trigrams(token)
				sentencetgs.append(c)

   		#-----------------Constructor of Sentence Class-----------------
			def __init__(self,sents,sentenceugs,sentencebgs,sentencetgs):
				self.sentenceugs = sentenceugs
				self.sentencebgs = sentencebgs
				self.sentencetgs = sentencetgs
				self.sents = sents
	   #-----------------Diplaying All the Sentence of the Given Text with Corresponding Unigram,Bigram and Trigram---------------
			def display(self):
				for i in range(0,len(sents)):
					print sents[i],sentenceugs[i],sentencebgs[i],sentencetgs[i]
					print "\n"



			#---------------Counting unigrams-----------------
			l1.extend([item for sublist in sentenceugs for item in sublist])
			counts1.append( Counter(l1))



			#---------------Counting bigrams-----------------
			l2 = [item for sublist in sentencebgs for item in sublist]
			counts2.append(Counter(l2))

			#---------------Counting trigrams-----------------
			l3 = [item for sublist in sentencetgs for item in sublist]
			counts3.append(Counter(l3))

			lsword =""
			for se in sentencebgs:
				for word in se:
					Sentset.add(word)

Sorted_Senttest = sorted(Sentset)
numConver = NumConver(Sorted_Senttest)
print counts2

temp2= {}
counts2_numeric = []
for co in counts2:
	for c in co:
		temp2[numConver.getNumericFromString(c)] = co[c]
	counts2_numeric.append(temp2)
	temp2 = {}



print counts2
print counts2_numeric

sent_obj = Sentence([],[],[],[])

#-------------The variable counts2 contains the bigrams.... each_text containts elements of the previously declared list "text".....
#-------------The variable ngram contains one single ngram for example ("machine","learning")---------------------
#There are several definitions of TF.Here assumed definition is:TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
bigramcount=len(counts2)
print "forming the dictionary..."
templist_for_ngrams=[]
templist_for_counts=[]
templist_for_counts_tfidf=[]
bigram_dict={}
bigram_dict_tfidf={}

sparse_key = []
sparse_value = []
Row_sparse_key = []
Row_sparse_value = []

for each_text in counts2:
	countx=0
	sparse_key = []
	sparse_value = []
	logsum = 0

	for ngram in each_text:
		if each_text[ngram] == 0:
			continue
		else:
			sparse_key.append(numConver.getNumericFromString(ngram))
			sparse_value.append(math.log(each_text[ngram]+1))

		templist_for_counts=[]# This is the list for making "value" part of the dictionary to be formed
		templist_for_counts_tfidf=[]
		flag=0
		for each_text2 in counts2:

			templist_for_counts.append(math.log(each_text2[ngram]+1))
			if each_text2[ngram]>0:
				flag=flag+1

		idf= 1/float(flag)

		sval = sum(templist_for_counts)
		for i in range(0,len(templist_for_counts)):
			templist_for_counts[i] /= sval

		templist_for_counts.append(idf)
		for i in range(0, len(templist_for_counts)-1):
			templist_for_counts_tfidf.append((idf) *(0.75*templist_for_counts[i]+0.125))
		strngram = " ".join((ngram))
		bigram_dict[strngram]=templist_for_counts# Finally this is the dictionary for bigram
		bigram_dict_tfidf[strngram]=templist_for_counts_tfidf
	logsum = sum(sparse_value)
	n_sparse_value = []

	#Here the logarithmic 
	#


	for svale in sparse_value:
	   n_sparse_value.append(0.5*0.25 + 0.75*(svale/logsum))
Rank_dict ={}
cs = 1
for key, value in bigram_dict_tfidf.items():
	s = sum(value)
	Rank_dict[key] = s
st = sorted(Rank_dict.items(),key = operator.itemgetter(1),reverse = True)
b = open('output_ranking.txt', 'w+')
b.write("")
b.close()
#c = open('output-tfidf21.txt', 'a')
b = open('output_ranking.txt', 'a')
for key, value in st:
	cs += 1
#	c.write(str(key) +"	   " +str(value)+"\n")
	b.write(str(key)+"\n")
b.close()
#c.close()
print("--- %s seconds ---" % (time.time() - start_time))
