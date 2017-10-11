import numpy as np
import pandas as pd
from nltk.corpus import reuters
import text_clean as tc

# quick summary of the reuters corpus
print("$$$ The reuters corpus has {} tags".format(len(reuters.categories())))
print("$$$ The reuters corpus has {} documents".format(len(reuters.fileids())))

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
print(f"The following categories are selected for the analysis: \
      {category_filter}")

# select fileid with the category filter
doc_list = np.array(reuters.fileids(category_filter))
doc_list = doc_list[doc_list != 'training/3267']

test_doc = doc_list[['test' in x for x in doc_list]]
train_doc = doc_list[['training' in x for x in doc_list]]

test_corpus = [" ".join([t for t in reuters.words(test_doc[t])])
               for t in range(len(test_doc))]
train_corpus = [" ".join([t for t in reuters.words(train_doc[t])])
                for t in range(len(train_doc))]

# create clean corpus for word2vec approach
test_clean_string = tc.clean_corpus(test_corpus)
train_clean_string = tc.clean_corpus(train_corpus)
print(test_clean_string[0])
# create clean corpus for bow approach
test_clean_token = tc.clean_corpus(test_corpus, string_line=False)
train_clean_token = tc.clean_corpus(train_corpus, string_line=False)
print(test_clean_token[0])
# quick look at the word frequency
test_bow, test_word_freq = tc.get_bow(test_clean_token)
train_bow, train_word_freq = tc.get_bow(train_clean_token)
