
## Introduction

In this post, I will showcase the steps I took to create a continuous vector space based on the corpora included in the famous [Reuters-21578 dataset](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection) (hereafter 'reusters dataset'). The reuters dataset is a tagged text corpora with news excerpts from Reuters newswire in 1987. Although the contents of the news is somewhat outdated, the topic labels provided in this dataset is widely used as a benchmark for supervised learning tasks that involve natural language processing (NLP).

[Word2Vec](https://en.wikipedia.org/wiki/Word2vec), among other word embedding models, is a neural network model that aims to generate numerical feature vectors for words in a way that maintains the relative meanings of the words in the mapped vectors. It provides a reliable method to reduce the feature dimensionality of the data, which is the biggest challenge for traditional NLP models such as the Bag of Words (BOW) model.

[BOW](https://en.wikipedia.org/wiki/Bag-of-words_model), on the other hand, is the traditional and established approach in text mining. The model deconstructs text into a list of words with either frequency or dummy checklist, hence creating a "bag of words" as a result. Although this approach is widely used as the beginner model for text-related projects, it has significant drawbacks:

* The sequencing and relative positioning of the words are largely ignored
* The resulted features are sparse, making it harder to build predictive models with


## Load Reuters Dataset

Before applying the models, I will generate a dataframe with the corpora. The first step is to download the reuters data and check what's inside.


```python
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# quick summary of the reuters corpus
print(">>> The reuters corpus has {} tags".format(len(reuters.categories())))
print(">>> The reuters corpus has {} documents".format(len(reuters.fileids())))
```

    >>> The reuters corpus has 90 tags
    >>> The reuters corpus has 10788 documents


The `reuters` from `nltk` comes with its specific sets of methods, making it hard to select and filter desired corpora. For my purpose in this report, I will only select 2 popular tags out of the 90 tags included in this corpora. Therefore, my next step is to generate a frequency table that summarizes the tags.


```python
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
```

       categories  file_count
    21       earn        3964
    0         acq        2369
    46   money-fx         717
    26      grain         582
    17      crude         578


I decide to chose the **second and third** tags on this top tags list, since the first **earn** tag is most likely the highly-standarized news pieces with earnings reports.


```python
# Select documents that only contains top two labels with most documents
cat_start = 1
cat_end = 2
category_filter = df.iloc[cat_start:cat_end + 1, 0].values.tolist()
print(f">>> The following categories are selected for the analysis: \
      {category_filter}")
```

    >>> The following categories are selected for the analysis:       ['acq', 'money-fx']


I then apply my tag filter on the text corpora to select text that have either the `acq` tag or the `money-fx` tag. The reuters data comes with a split of training and testing itself, so I will stick with its original split.


```python
# select fileid with the category filter
doc_list = np.array(reuters.fileids(category_filter))
doc_list = doc_list[doc_list != 'training/3267']

test_doc = doc_list[['test' in x for x in doc_list]]
print(">>> test_doc is created with following document names: {} ...".format(test_doc[0:5]))
train_doc = doc_list[['training' in x for x in doc_list]]
print(">>> train_doc is created with following document names: {} ...".format(train_doc[0:5]))

test_corpus = [" ".join([t for t in reuters.words(test_doc[t])])
               for t in range(len(test_doc))]
print(">>> test_corpus is created, the first line is: {} ...".format(test_corpus[0][:100]))
train_corpus = [" ".join([t for t in reuters.words(train_doc[t])])
                for t in range(len(train_doc))]
print(">>> train_corpus is created, the first line is: {} ...".format(train_corpus[0][:100]))
```

    >>> test_doc is created with following document names: ['test/14843' 'test/14849' 'test/14852' 'test/14861' 'test/14865'] ...
    >>> train_doc is created with following document names: ['training/10' 'training/1000' 'training/10005' 'training/10018'
     'training/10025'] ...
    >>> test_corpus is created, the first line is: SUMITOMO BANK AIMS AT QUICK RECOVERY FROM MERGER Sumitomo Bank Ltd & lt ; SUMI . T > is certain to l ...
    >>> train_corpus is created, the first line is: COMPUTER TERMINAL SYSTEMS & lt ; CPML > COMPLETES SALE Computer Terminal Systems Inc said it has com ...


Now that I have the corpora in place, it's time to clean up the text to reduce the noice from non-text elements. I have stored my text-cleaning functions in another module called `text_clean.py` and I will import it to clean up the text. My module document as well as all codes related to this project can be found at my GitHub Repo [here](https://github.com/ElvinOuyang/reuters-w2v-practice).


```python
import text_clean as tc

# create clean corpus for word2vec approach
test_clean_string = tc.clean_corpus(test_corpus)
train_clean_string = tc.clean_corpus(train_corpus)
print('>>> The first few words from cleaned test_clean_string is: {}'.format(test_clean_string[0][:100]))
print('>>> The first few words from cleaned train_clean_string is: {}'.format(train_clean_string[0][:100]))
```

    >>>> response cleaning initiated
    >>>> cleaning response #500 out of 898
    >>>> response cleaning initiated
    >>>> cleaning response #500 out of 2186
    >>>> cleaning response #1000 out of 2186
    >>>> cleaning response #1500 out of 2186
    >>>> cleaning response #2000 out of 2186
    >>> The first few words from cleaned test_clean_string is: sumitomo bank aim quick recovery merger sumitomo bank ltd lt sumi certain lose status japan profitab
    >>> The first few words from cleaned train_clean_string is: computer terminal systems lt cpml complete sale computer terminal systems inc say complete sale shar


## Glimpse of BOW model
After the text is cleaned, I can now apply the BOW model on the corpora. The BOW is basically a frequency `Counter`, so I have written a function to get BOW from the corpora in my `text_clean.py` module.


```python
# create clean corpus for bow approach
test_clean_token = tc.clean_corpus(test_corpus, string_line=False)
train_clean_token = tc.clean_corpus(train_corpus, string_line=False)
# quick look at the word frequency
test_bow, test_word_freq = tc.get_bow(test_clean_token)
train_bow, train_word_freq = tc.get_bow(train_clean_token)
```

    >>>> response cleaning initiated
    >>>> cleaning response #500 out of 898
    >>>> response cleaning initiated
    >>>> cleaning response #500 out of 2186
    >>>> cleaning response #1000 out of 2186
    >>>> cleaning response #1500 out of 2186
    >>>> cleaning response #2000 out of 2186
    This corpus has 7126 key words, and the 10 most frequent words are: [('say', 2976), ('lt', 1112), ('share', 1067), ('dlrs', 921), ('company', 886), ('pct', 758), ('mln', 755), ('inc', 637), ('bank', 505), ('corp', 500)]
    This corpus has 11042 key words, and the 10 most frequent words are: [('say', 7388), ('lt', 2802), ('share', 2306), ('dlrs', 2247), ('mln', 2172), ('pct', 2051), ('bank', 1983), ('company', 1942), ('inc', 1469), ('u', 1291)]


The BOW is a frequency table that records the word frequencies for each news piece in the corpora. If I turn it into a table view, it is clear that the most of the columns will be `NaN` due to the sparse nature of a BOW model. In fact, for the few rows of the `test_bow` table, all displayed cells below are NaN. Tables like this will make most of the data science models inprecise when we conduct predictive analysis.


```python
print(pd.DataFrame(test_bow).head())
```

       aabex  aame  aar  ab  abandon  abate  abatement  abboud  abegglen  abeles  \
    0    NaN   NaN  NaN NaN      NaN    NaN        NaN     NaN       NaN     NaN   
    1    NaN   NaN  NaN NaN      NaN    NaN        NaN     NaN       NaN     NaN   
    2    NaN   NaN  NaN NaN      NaN    NaN        NaN     NaN       NaN     NaN   
    3    NaN   NaN  NaN NaN      NaN    NaN        NaN     NaN       NaN     NaN   
    4    NaN   NaN  NaN NaN      NaN    NaN        NaN     NaN       NaN     NaN   
    
         ...     zellerbach  zenex  zinn  zoete  zond  zondervan  zone  zoran  \
    0    ...            NaN    NaN   NaN    NaN   NaN        NaN   NaN    NaN   
    1    ...            NaN    NaN   NaN    NaN   NaN        NaN   NaN    NaN   
    2    ...            NaN    NaN   NaN    NaN   NaN        NaN   NaN    NaN   
    3    ...            NaN    NaN   NaN    NaN   NaN        NaN   NaN    NaN   
    4    ...            NaN    NaN   NaN    NaN   NaN        NaN   NaN    NaN   
    
       zurich  zwermann  
    0     NaN       NaN  
    1     NaN       NaN  
    2     NaN       NaN  
    3     NaN       NaN  
    4     NaN       NaN  
    
    [5 rows x 7126 columns]


The typical way to reduce the noise in BOW models is to use dimensionality reduction methods, such as [Latent Semantic Indexing (LSA)](https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing), [Random Projections (RP)](http://users.ics.aalto.fi/ella/publications/randproj_kdd.pdf), [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), or [Hierachical Dirichlet Process (HDP)](http://proceedings.mlr.press/v15/wang11a/wang11a.pdf). 

## Glimpse of Word2Vec Model
The alternative, of course, is to use the famous word2vec algorithm to generate continous numeric vectors. I will use `gensim` package to conduct this task. After I train the model with the reuters corpora, I get a dictionary that maps a numeric vector to each word that appears in the corpora.


```python
import gensim
import logging

# set up logging for gensim training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.Word2Vec(train_clean_token, min_count=1, workers=2)
```

    2017-10-17 10:41:29,134 : INFO : collecting all words and their counts
    2017-10-17 10:41:29,135 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2017-10-17 10:41:29,197 : INFO : collected 11042 word types from a corpus of 191770 raw words and 2186 sentences
    2017-10-17 10:41:29,198 : INFO : Loading a fresh vocabulary
    2017-10-17 10:41:29,235 : INFO : min_count=1 retains 11042 unique words (100% of original 11042, drops 0)
    2017-10-17 10:41:29,236 : INFO : min_count=1 leaves 191770 word corpus (100% of original 191770, drops 0)
    2017-10-17 10:41:29,295 : INFO : deleting the raw counts dictionary of 11042 items
    2017-10-17 10:41:29,297 : INFO : sample=0.001 downsamples 40 most-common words
    2017-10-17 10:41:29,298 : INFO : downsampling leaves estimated 169461 word corpus (88.4% of prior 191770)
    2017-10-17 10:41:29,300 : INFO : estimated required memory for 11042 words and 100 dimensions: 14354600 bytes
    2017-10-17 10:41:29,375 : INFO : resetting layer weights
    2017-10-17 10:41:29,533 : INFO : training model with 2 workers on 11042 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2017-10-17 10:41:30,539 : INFO : PROGRESS: at 80.76% examples, 681629 words/s, in_qsize 3, out_qsize 0
    2017-10-17 10:41:30,878 : INFO : worker thread finished; awaiting finish of 1 more threads
    2017-10-17 10:41:30,887 : INFO : worker thread finished; awaiting finish of 0 more threads
    2017-10-17 10:41:30,888 : INFO : training on 958850 raw words (846962 effective words) took 1.4s, 626379 effective words/s


With the model, I can get the numeric representation of any word that's included in the model's dictionary, and even quickly calculate the "similarity" between two words.


```python
print(">>> Printing the vector of 'inc': {} ...".format(model['inc'][:10]))
print(">>> Printing the similarity between 'inc' and 'love': {}"\
      .format(model.wv.similarity('inc', 'love')))
print(">>> Printing the similarity between 'inc' and 'company': {}"\
      .format(model.wv.similarity('inc', 'company')))
```

    >>> Printing the vector of 'inc': [ 0.78641725 -0.81131667  2.96111465 -1.71037292  1.00312221  0.0688597
      0.13069828  0.99924785 -0.35323438  0.14066967] ...
    >>> Printing the similarity between 'inc' and 'love': 0.8529129076656723
    >>> Printing the similarity between 'inc' and 'company': 0.9254433388270853


*inc* and *company* appears to be more similar than *inc* and *love*. After I get the vector representation of each word, I can calculate the vector representation of each news piece by simply getting the vector mean of all words included in that news piece. There certainly are more advanced aggregation methods, but for this practice I will use the most straightforward way. Another thing worth noticing is that I need to assign an all-zero vector if a word is not included in the dictionary of the Word2Vec model.

I also created a module for the matrix aggregation in my GitHub repo [here](https://github.com/ElvinOuyang/reuters-w2v-practice); so I simply import the function `get_doc_matrix` and calculate the document vectors.


```python
from w2v_cal import get_doc_matrix

test_matrix = get_doc_matrix(model, test_clean_token)
train_matrix = get_doc_matrix(model, train_clean_token)
```

Be noted that my `word2vec` model was trained with the `train_clean_token` corpus, but I use the model on the `test_clean_token` here. I did it on purpose because in real-life situation, you won't know your test set when you train the model. If you need to evaluate how accurate the model can be, you need to seperate the train and test even on the word embedding stage.

A work-around that I've thought of is to update the `word2vec` model on the fly by feeding it the new corpora that come with the new documents: in this way I only update the model with the information that I know of - the test document I'm dealing with and all the train documents. I will further elaborate on this idea in my future posts about word embedding applications.


```python
print(pd.DataFrame(test_matrix).head())
```

             0         1         2         3         4         5         6   \
    0  0.147100 -0.175438  0.888369 -0.069528  0.033833 -0.013307  0.233525   
    1  0.042170 -0.163114  0.978680  0.227916 -0.142440 -0.083064  0.235953   
    2  0.155263 -0.208363  1.048186 -0.157641  0.127218 -0.066839  0.140428   
    3 -0.023057 -0.217250  1.123171  0.460213 -0.346473 -0.105230  0.137149   
    4  0.183727 -0.297932  1.154441 -0.333167  0.194415 -0.019180  0.009379   
    
             7         8         9     ...           90        91        92  \
    0  0.274196 -0.175343 -0.073920    ...     0.069914 -0.158157  0.241313   
    1  0.349826 -0.203713 -0.163635    ...     0.014871 -0.315029  0.215814   
    2  0.388008 -0.176918 -0.088959    ...     0.181733 -0.196467  0.211149   
    3  0.438802 -0.264249 -0.317894    ...    -0.017868 -0.349207  0.047567   
    4  0.449627 -0.152873 -0.030410    ...     0.272388 -0.197652  0.166376   
    
             93        94        95        96        97        98        99  
    0 -0.301072 -0.613152 -0.001587  0.609080  0.174361 -0.124058 -0.203552  
    1 -0.306695 -0.832693  0.124513  0.852931  0.039255 -0.264621 -0.344080  
    2 -0.378385 -0.608279 -0.048504  0.716636  0.140785 -0.093196 -0.194501  
    3 -0.359221 -0.989569  0.082096  1.062137 -0.133572 -0.353757 -0.346747  
    4 -0.503684 -0.586876 -0.133392  0.807204  0.136802 -0.059075 -0.122065  
    
    [5 rows x 100 columns]


A glimpse of the resulted word matrix indicates that the dataset is now filled with continous numeric values. I can now use more models to predict the labels of these news pieces. I will test the effectiveness of these two models in my upcoming posts.
