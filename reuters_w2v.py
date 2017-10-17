import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import reuters
import gensim
import logging
import text_clean as tc
import w2v_cal as wc

# quick summary of the reuters corpus
print(">>> The reuters corpus has {} tags".format(len(reuters.categories())))
print(">>> The reuters corpus has {} documents".format(len(reuters.fileids())))

# create counter to summarize
categories = []
file_count = []

# count each tag's number of documents
for i in reuters.categories():
    """print("$ There are {} documents included in topic \"{}\""
          .format(len(reuters.fileids(i)), i))"""
    file_count.append(len(reuters.fileids(i)))
    categories.append(i)

# create a dataframe out of the counts
df = pd.DataFrame(
    {'categories': categories, "file_count": file_count}) \
    .sort_values('file_count', ascending=False)
print(df.head())

# Select documents that only contains top two labels with most documents
cat_start = 1
cat_end = 2
category_filter = df.iloc[cat_start:cat_end + 1, 0].values.tolist()
print(f">>> The following categories are selected for the analysis: \
      {category_filter}")

# select fileid with the category filter
doc_list = np.array(reuters.fileids(category_filter))
doc_list = doc_list[doc_list != 'training/3267']

test_doc = doc_list[['test' in x for x in doc_list]]
print(">>> test_doc is created with following document names: {} ...".
      format(test_doc[0:5]))
train_doc = doc_list[['training' in x for x in doc_list]]
print(">>> train_doc is created with following document names: {} ...".
      format(train_doc[0:5]))

test_corpus = [" ".join([t for t in reuters.words(test_doc[t])])
               for t in range(len(test_doc))]
print(">>> test_corpus is created, the first line is: {} ...".
      format(test_corpus[0][:100]))
train_corpus = [" ".join([t for t in reuters.words(train_doc[t])])
                for t in range(len(train_doc))]
print(">>> train_corpus is created, the first line is: {} ...".
      format(train_corpus[0][:100]))


# create clean corpus for word2vec approach
test_clean_string = tc.clean_corpus(test_corpus)
train_clean_string = tc.clean_corpus(train_corpus)
print('>>> The first few words from cleaned test_clean_string is: {}'.
      format(test_clean_string[0][:100]))
print('>>> The first few words from cleaned train_clean_string is: {}'.
      format(train_clean_string[0][:100]))

# create clean corpus for bow approach
test_clean_token = tc.clean_corpus(test_corpus, string_line=False)
train_clean_token = tc.clean_corpus(train_corpus, string_line=False)

# set up logging for gensim training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

model = gensim.models.Word2Vec(train_clean_token, min_count=1, workers=2)

print(">>> Printing the vector of 'inc': {} ...".format(model['inc'][:10]))
print(">>> Printing the similarity between 'inc' and 'love': {}"
      .format(model.wv.similarity('inc', 'love')))
print(">>> Printing the similarity between 'inc' and 'company': {}"
      .format(model.wv.similarity('inc', 'company')))


test_matrix = wc.get_doc_matrix(model, test_clean_token)
train_matrix = wc.get_doc_matrix(model, train_clean_token)

print(pd.DataFrame(test_matrix).head())
