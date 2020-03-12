"""
    1. Build a dictionary from the entire corpus of filings
    2. Build word2vec model to embed all words from the dictionary
"""

import os
import re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
LEMMATIZER = WordNetLemmatizer()
PUNCTUATION = '.,:;``\'\'""!?()--=$&%'
DIGITS = '0123456789'
PROHIBITED_TOKENS = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'vix', 'x', '\'s']

PROCESSED_DATA_DIR = os.path.join('..', 'data', 'relevant_text')
FILING_INDEX_CSV = os.path.join('..', 'data', 'new_full_filing_index.csv')
FIRST_TEST_DATE = '2011-01-01'


################################################
##  Misc. utilities
################################################
def file_from_row(row):
    """ Return path to processed file from filing index row """

    name = row['local_filename'].split('.')[0]
    return os.path.join(PROCESSED_DATA_DIR, f"{ name }_{ row['Form Type'] }_output.txt")

def clean_tokens(text, lemmatizer=LEMMATIZER):
    """ Convert text into list of clean (word) tokens """

    # Remove non-ASCII codes like \x93
    text = text.encode('ascii', 'ignore').decode('ascii', 'ignore')

    # Clear some troublesome special chars
    text = re.sub(r'[\-=_%/\[\]@]', ' ', text)

    # Lower-case, lemmatize, remove punctuation and stopwords
    tokens = [
        LEMMATIZER.lemmatize(t, pos='v') for t in nltk.word_tokenize(text.lower())
        if (
            t not in PUNCTUATION    # Prohibit punctuation
            and t not in STOPWORDS  # Prohibit stopwords
            and not any((d in t for d in DIGITS))   # Prohibit tokens with numbers
            and len(t) > 1          # Prohibit length-1 tokens
        )
    ]

    # Clear some special-case tokens
    tokens = [t for t in tokens if t not in PROHIBITED_TOKENS]

    return tokens

def filing_word_counts(file, lemmatizer=LEMMATIZER):
    """ Return dict of the form {'word': count} """

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    words = clean_tokens(text, lemmatizer)
    counts = {}
    for t in words:
        if any((d in t for d in '0123456789<>[]\{\}%@#^*~+$=_\\')): 
            continue
        if t in counts:
            counts[t] += 1
        else:
            counts[t] = 1
    return counts

def word_counts_from_text(text):
    """ Copy of filing_word_counts but input is a string, not a file """

    words = clean_tokens(text)
    counts = {}
    for t in words:
        if any((d in t for d in '0123456789<>[]\{\}%@#^*~+$=_\\')): 
            continue
        if t in counts:
            counts[t] += 1
        else:
            counts[t] = 1
    return counts

def save_word_freqs(sample_size=5000):
    """ Save a dictionary of word frequencies (for words in the dictionary) """

    print('Load filings, dictionary')
    filings = pd.read_csv(FILING_INDEX_CSV, index_col=0, parse_dates=['Date Filed'])
    filings = filings.loc[filings['Date Filed'] < FIRST_TEST_DATE]
    filings = filings.sample(n=sample_size, replace=False, random_state=314159)
    dictionary = load_dictionary()
    totalCounts = {k: 0 for k in dictionary}
    numFilings = len(filings)

    i = 0
    for idx, row in filings.iterrows():
        i += 1
        file = file_from_row(row)
        if i % 100 == 0:
            print(f'--Process file { i } of { numFilings }')
        try:
            counts = filing_word_counts(file)
            for word, count in counts.items():
                if word in totalCounts:
                    totalCounts[word] += count
                else:
                    totalCounts['<UNK>'] += count
            # print('----Added word counts to total')
        except Exception as e:
            print(f'----Exception: {e}')
    totalSum = sum(totalCounts.values())
    print(f'Total number of tokens in sample: { totalSum }')

    totalSum = float(totalSum)
    wordFreqs = {k: v/totalSum for k, v in totalCounts.items()}
    print('Saving word_freqs.json')
    with open('word_freqs.json', 'w') as f:
        json.dump(wordFreqs, f)
    return wordFreqs

def load_word_freqs():
    with open('word_freqs.json', 'r') as f:
        return json.load(f)



################################################
##  Build the dictionary
################################################
def build_dictionary(sample_size=5000):
    """ Save dictionary mapping most common words in train data sample to index """

    print('Computing word counts from sample')
    filings = pd.read_csv(FILING_INDEX_CSV, index_col=0, parse_dates=['Date Filed'])
    filings = filings.loc[filings['Date Filed'] < FIRST_TEST_DATE]
    filings = filings.sample(n=sample_size, replace=False, random_state=314159)
    totalCounts = {}
    numFilings = len(filings)
    i = 0
    for idx, row in filings.iterrows():
        i += 1
        file = file_from_row(row)
        print(f'--Getting word counts for file { i } of { numFilings }')
        try:
            counts = filing_word_counts(file)
            for word, count in counts.items():
                if word in totalCounts:
                    totalCounts[word] += count
                else:
                    totalCounts[word] = count
            print('----Added word counts to total')
        except Exception as e:
            print(f'----Exception: {e}')
    sortedCounts = sorted(list(totalCounts.items()), key=lambda t: t[1], reverse=True)
    print(f'Total words in sample: { len(sortedCounts) }')
    dictionaryLength = 20000
    dictionary = {word: i for i, (word, count) in enumerate(sortedCounts[:dictionaryLength])}
    dictionary['<UNK>'] = dictionaryLength
    print('Saving dictionary')
    with open('dictionary.json', 'w') as f:
        json.dump(dictionary, f)
    
    return dictionary
    
def load_dictionary():
    with open('dictionary.json', 'r') as f:
        return json.load(f)



################################################
##  Main
################################################
if __name__ == '__main__':

    # Load a filing and compute its word counts
    """
    filings = pd.read_csv('..\\data\\sp500_sample.csv', index_col=0)
    file = file_from_row(filings.iloc[0])
    print(f'Getting word counts for { file }')
    try:
        counts = filing_word_counts(file)
        sortedCounts = sorted(list(counts.items()), key=lambda t: t[1], reverse=True)
        for word, count in sortedCounts:
            print(f'{ word }: { count }')
    except Exception as e:
        print('File word count exception:')
        print(e)
    """

    # Load 100 filings and compute total word counts
    """
    filings = pd.read_csv('..\\data\\sp500_sample.csv', index_col=0)
    totalCounts = {}
    for i in range(1000):
        file = file_from_row(filings.iloc[i])
        print(f'Getting word counts for file { i+1 }')
        try:
            counts = filing_word_counts(file)
            for word, count in counts.items():
                if word in totalCounts:
                    totalCounts[word] += count
                else:
                    totalCounts[word] = count
            print('--Added word counts to total')
        except Exception as e:
            print(f'--Exception: {e}')
    sortedCounts = sorted(list(totalCounts.items()), key=lambda t: t[1], reverse=True)
    print('\nDone. 100 most common words:')
    for word, count in sortedCounts[:100]:
        print(f'{ word }: { count }')
    """

    # Build the dictionary
    # build_dictionary()

    # Generate word frequencies for dictionary words
    save_word_freqs(sample_size=5000)

    # Test tokenization
    # filings = pd.read_csv('..\\data\\sp500_sample.csv', index_col=0)
    # with open(file_from_row(filings.iloc[0]), 'r', encoding='utf-8') as f:
    #     text = f.read()
    # print(clean_tokens(text))