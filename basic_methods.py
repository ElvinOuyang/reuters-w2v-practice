# preparation
from __future__ import division
import nltk
from nltk.book import *

'Searching Text'
# .concordance() will display context of each searched words
text1.concordance("monstrous")
text2.concordance("affection")
text3.concordance("lived")
print("\n")

# .similar() will return words that appear in similar context
text1.similar("monstrous")
print("\n")
text2.similar("affection")
print("\n")
text3.similar("lived")

# .commmon_contexts() will return the contexts shared by two or more words,
# separated with commas
text2.common_contexts(["monstrous","very"])

# display the words' locations with dispersion plots
text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])

'Tokens, word types, and lexical richness'
print(sorted(set(text3)))
# a list of all word types included in text3
# set() returns unique token occurrence within a text

print(len(set(text3)))
# 2789 word types

print(len(text3) / len(set(text3)))
# Calculates the "lexical richness" of the text, namely how many times each word is used in the text

print(text3.count("smote"))
# Calculates how many times a word appear in a text
print(100 * text5.count('lol') / len(text5))
# Calculates how much a certain word takes in the whole text string


def lexical_diversity(text):
    # A function that returns average usage for each word in a piece of text
    return len(text) / len(set(text))


def percentage(word, full_text):
    # A function that returns the percentage of a word in a text
    return 100 * full_text.count(word) / len(full_text)

# Use "+" operators for concatenation
