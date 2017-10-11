import numpy as np
import pandas as pd
from nltk.corpus import reuters
import text_clean as tc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
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

# Select documents that only contains top two labels with most documents
cat_start = 0
cat_end = cat_start + 1
category_filter = df.iloc[cat_start:cat_end + 1, 0].values.tolist()
print(f"The following categories are selected for the analysis: \
      {category_filter}")

# select fileid with the category filter
doc_list = np.array(reuters.fileids(category_filter))
# TODO: create a function to exclude overlapped documents
# doc_list = doc_list[doc_list != 'training/3267']

test_doc = doc_list[['test' in x for x in doc_list]]
train_doc = doc_list[['training' in x for x in doc_list]]

test_corpus = [" ".join([t for t in reuters.words(test_doc[t])])
               for t in range(len(test_doc))]
train_corpus = [" ".join([t for t in reuters.words(train_doc[t])])
                for t in range(len(train_doc))]

# create clean corpus
test_clean_string = tc.clean_corpus(test_corpus)
train_clean_string = tc.clean_corpus(train_corpus)
# clean_string = train_clean_string + test_clean_string
# vectorize the corpus
tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                   tokenizer=None,
                                   preprocessor=None,
                                   stop_words=None,
                                   max_features=5000,
                                   ngram_range=(1, 2),
                                   min_df=10)
# TODO: if the vectorize-update approach doesn't work, use this approach
"""
dtm_tfidf = \
    tfidf_vectorizer.fit_transform(clean_string).toarray()
print(f"DTM tfidf is created with shape {dtm_tfidf.shape}.")
train_dtm_tfidf = dtm_tfidf[:len(train_clean_string), :]
print(f"training DTM has shape {train_dtm_tfidf.shape}")
test_dtm_tfidf = dtm_tfidf[len(train_clean_string):, :]
print(f"test DTM has shape {test_dtm_tfidf.shape}")
"""
# vectorize-update approach
# create the DTM dictionary based on the training corpus
train_dtm_tfidf = \
    tfidf_vectorizer.fit_transform(train_clean_string).toarray()
print(f"training DTM tfidf is created with shape {train_dtm_tfidf.shape}.")
# create the DTM for test using the existing dictionary
test_dtm_tfidf = \
    tfidf_vectorizer.transform(test_clean_string).toarray()
print(f"test DTM has shape {test_dtm_tfidf.shape}")

# create the target tag variable for the training
test_cat = list(map(lambda x: reuters.categories(x), test_doc))
train_cat = list(map(lambda x: reuters.categories(x), train_doc))
test_target = np.array(list(map(lambda x: category_filter[0] in x, test_cat)))
train_target = np.array(
    list(map(lambda x: category_filter[0] in x, train_cat)))

# train the bow model with random forest
print("Training the random forest...")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_dtm_tfidf, train_target)
test_predict = forest.predict(test_dtm_tfidf)
print(confusion_matrix(test_target, test_predict))
print(classification_report(test_target, test_predict,
      target_names=[category_filter[0], category_filter[1]]))

# train the bow model with naive bayes
print("Training the naive bayes...")
nb_clf = MultinomialNB()
nb_clf = nb_clf.fit(train_dtm_tfidf, train_target)
test_predict = nb_clf.predict(test_dtm_tfidf)
print(confusion_matrix(test_target, test_predict))
print(classification_report(test_target, test_predict,
      target_names=[category_filter[0], category_filter[1]]))

# train the bow model with naive bayes
print("Training the SVM...")
svm_clf = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42)
svm_clf = svm_clf.fit(train_dtm_tfidf, train_target)
test_predict = svm_clf.predict(test_dtm_tfidf)
print(confusion_matrix(test_target, test_predict))
print(classification_report(test_target, test_predict,
      target_names=[category_filter[0], category_filter[1]]))
