
import re
import string
from pathlib import Path

import en_core_web_sm
import nltk
import PyPDF2
import spacy
import wikipedia

from gensim.parsing.preprocessing import remove_stopwords
from gensim.summarization import keywords

from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from rake_nltk import Rake

#https://www.ranks.nl/stopwords

def get_PDF(filepath):
    doc = open(filepath, 'rb')
    reader = PyPDF2.PdfFileReader(doc)

    text = []
    for page in reader.pages:
        text.append(page.extractText())

    return ' '.join(text)

# def remove_mystopwords(sentence):
#     tokens = sentence.split(" ")
#     tokens_filtered= [word for word in text_tokens if not word in my_stopwords]
#     return (" ").join(tokens_filtered)

def lemmatise_text(text):
    sp = en_core_web_sm.load()

    lem_text = []
    sp_text = sp(text)

    for word in sp_text:
        lem_text.append(word.lemma_)

    return " ".join(lem_text)


def stem_text(text, NLTKstemmer, generate_stem_dict=True):
    # porter = PorterStemmer()
    token_words = word_tokenize(text)

    stem_dict = {}
    stem_sentence = []
    for word in token_words:
        stemmed_word = NLTKstemmer.stem(word)
        stem_sentence.append(stemmed_word)

        if generate_stem_dict is True:
            # Generates dictionary of stemmed word to real words
            if stemmed_word not in stem_dict:
                stem_dict[stemmed_word] = set()
            stem_dict[stemmed_word].add(word)

    stemmed_text = " ".join(stem_sentence)

    if generate_stem_dict is True:
        newDict = {}
        for k, v in stem_dict.items():
            if len(v) > 1:
                newDict[k] = v
        return stemmed_text, newDict
    else:
        return stemmed_text


def remove_numbers(text):
    new_text = []
    for word in word_tokenize(text):
        if word.isdigit() is False:
            new_text.append(word)
    # ''.join(c for c in text if not c.isdigit())
    return ' '.join(new_text)

def remove_carriage_returns(text):
    return text.replace('\n', ' ')

def remove_special_chars(text, special_chars):

    for char in special_chars:
        text = text.replace(char, ' ')

    return text

def pmi(text, nbest=10, freq=5):
    gram = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(word_tokenize(text))
    finder.apply_freq_filter(freq)
    return finder.nbest(gram.pmi, nbest)

def pos_text(text, POS_tags):
    sp = en_core_web_sm.load()
    new_text = []
    for token in sp(text):
        if token.pos_ in POS_tags:
            new_text.append(token.text)
    return ' '.join(new_text)

class FetchWiki:

    def __init__(self, page_title):
        self.pagetitle = page_title
        self.page = wikipedia.page(page_title)
        self.text = self.page.content


class TextRank:
    def __init__(self, text):
        self.text = text

    def getKeywords(self, top_n=10):
        kws = keywords(self.text).split('\n')[0:top_n]
        return kws

class TextRake:
    def __init__(self, text, max_phrase_length=4):
        self.text = text
        self.rake = Rake(max_length=max_phrase_length)

    def getKeywords(self, top_n=10):
        self.rake.extract_keywords_from_text(self.text)
        return self.rake.get_ranked_phrases()[0:top_n]

text_raw = get_PDF(r'C:\Users\hag67301\Documents\GitHub\keyword-extraction\C783 UAVs for managing assets.pdf')

text = text_raw
# p = FetchWiki('Coronavirus disease 2019')
# text = p.text

# Remove carriage returns
text = remove_carriage_returns(text)

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Remove special characters
text = remove_special_chars(text, '@#©™•£$')

# Lower case
text = str.lower(text)

# Remove numbers
text = remove_numbers(text)

# RAKE
text_rake = TextRake(text, max_phrase_length=2)
print('\n===== TEXT RAKE =====\n')
print(text_rake.getKeywords(top_n=100))

# =======
# Remove stopwords
text = remove_stopwords(text)

# POS
keep_tags = {"NOUN", "VERB", "ADJ", "ADP", "PROPN"}
text = pos_text(text, keep_tags)

# Lemma
# text = lemmatise_text(text)

# Stem
text, stemDict = stem_text(text, PorterStemmer(), generate_stem_dict=True)

# TEXT RANK
text_rank = TextRank(text)
print('\n===== TEXT RANK =====\n')
print(text_rank.getKeywords(top_n=100))

# PMI
pmi(text, nbest=50, freq=2)


####################
# s = 'Generally, Pythoners are very intelligent and work very pythonly and now they are pythoning their way to success.'
# s = 'run running runner runs ran'

# s_lem = lemmatise_text(s)
# print(set(word_tokenize(s_lem)))
# s_lem_stem = stem_text(s_lem, PorterStemmer())
# print(set(word_tokenize(s_lem_stem)))
