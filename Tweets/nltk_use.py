# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-

import gensim
#
word_path = '../File/GoogleNews-vectors-negative300.bin'
Word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(word_path,binary=True)
print(Word2VecModel.vector_size)
print(Word2VecModel['unfill'])


# from textblob import TextBlob
# from nltk.stem import SnowballStemmer
# from nltk.corpus import wordnet
# # 词干提取
# stemmer = SnowballStemmer("english") # Choose a languagestemmer.stem("countries") # Stem a word
# new_word = stemmer.stem("unfilled")
# print(new_word)
# B = TextBlob(new_word)
# print(B.correct())
# # 词形还原
# from nltk.stem import WordNetLemmatizer
# wnl = WordNetLemmatizer()
# print(wnl.lemmatize("unfilled",pos = wordnet.ADJ))



from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

sentence = 'Unfilled positions number several million countrywide.'

# 分词
tokens = word_tokenize(sentence)
# 获取单词词性
tagged_sent = pos_tag(tokens)
print(tagged_sent)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

wnl = WordNetLemmatizer()
lemmas_sent = []

for word,word_tag in tagged_sent:
    wordnet_pos = get_wordnet_pos(word_tag)
    # new_word = stemmer.stem(word)
    lemmas_sent.append(wnl.lemmatize(new_word,pos=wordnet_pos))
print(lemmas_sent)