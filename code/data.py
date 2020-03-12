"""
    Prepare datasets to train CBOW word2vec model
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
LEMMATIZER = WordNetLemmatizer()
import os
import sys
import json
sys.path.append(os.path.join('..', 'practice_qcf', 'analysis'))
from word_embed import clean_tokens
from model import vocabulary, inverse_vocabulary, vocab_index, index_to_word, cw_to_ci, ci_to_cw, word_freqs
import pandas as pd

PROCESSED_DATA_DIR = 'D:\\qcf_nlp\\practice_qcf\\data\\relevant_text'
FILING_INDEX_FNAME = 'D:\\qcf_nlp\\practice_qcf\\data\\new_full_filing_index.csv'
FIRST_TEST_DATE = '2011-01-01'


##############################################
##   Build full dataset from all filings
##############################################
def keep_word(word):
    """ Return bool indicating if word should be dropped
        based on probability computed from word frequency 
    """

    try:
        freq = word_freqs[word]
    except KeyError:
        freq = word_freqs['<UNK>']
    return np.random.random() <= (np.sqrt(freq/.001) + 1) * (.001/freq)

def filename_from_row(row):
    return row['local_filename'].split('.')[0] + '_' + row['Form Type'] + '_output.txt'

def dataset_from_filing(fname, subsample=False):
    """ Create a sample dataset from a single document 

        Args:
            fname (str)       -- the filename of the document
            subsample (bool)  -- if True, randomly delete words based on their frequency (Default: False)
        
        Returns:
            (np.array (4, N))   -- context words (input to model)
            (np.array (1, N))   -- target words
              where N is the number of data points generated from the document
              (ie number of locations for the sliding context window)
    """

    # print(f'Build dataset from { fname }')

    # Open, clean, tokenize an example document
    with open(os.path.join(PROCESSED_DATA_DIR, fname), encoding='utf-8') as f:
        exText = f.read()
    exTokens = clean_tokens(exText)

    if len(exTokens) < 100:
        # print('Skipping file; not enough tokens')
        return (None, None)

    # Subsample
    if subsample:
        exTokens = [t for t in exTokens if keep_word(t)]

    # Construct CBOW training data
    # Note: hard-coded context window of 5 here
    contextWords = [
        ([exTokens[i], exTokens[i+1], exTokens[i+3], exTokens[i+4]], exTokens[i+2])
        for i in range(len(exTokens) - 4)
    ]

    # Map tokens to indices in vocabulary
    contextIndices = [cw_to_ci(cw) for cw in contextWords]

    # Convert to numpy arrays
    contextArr = np.array([ci[0] for ci in contextIndices])
    targetArr  = np.array([ci[1] for ci in contextIndices])

    # print(f'Generated { contextArr.shape[0] } samples')

    return contextArr, targetArr

def build_full_dataset(filings, X_fname, y_fname, subsample=False):
    """ Build full dataset from a csv of filings """

    print('Build datasets')
    datasets = [
        dataset_from_filing(filename_from_row(row), subsample=subsample)
        for i, (idx, row) in enumerate(filings.iterrows())
    ]

    print('Remove empty returned datasets')
    # These are indicated by (None, None)
    datasets = [p for p in datasets if p[0] is not None]
    print(f'{ len(datasets) } datasets left to stack')

    print('Stack')
    X_full = np.vstack([p[0] for p in datasets])
    y_full = np.concatenate([p[1] for p in datasets])

    print(X_full.shape, y_full.shape)
    print(f'{ X_full.shape[0] } total samples')

    print(f'Saving to { X_fname }, { y_fname }')
    np.save(X_fname, X_full)
    np.save(y_fname, y_full)

    return X_full, y_full

def build_batched_dataset(file_identifier, start_batch=0, subsample=False):
    """ Build the full dataset in batches of 2000 files at a time """

    print('Load file index')
    filings = pd.read_csv(FILING_INDEX_FNAME, parse_dates=['Date Filed'])

    print('Select train data')
    filings = filings.loc[filings['Date Filed'] < FIRST_TEST_DATE]

    print('Save datasets in batches')

    ### Adjust this if theres an error or you stop partway through
    START_BATCH = start_batch

    batchSize = 2000
    batchCounter = START_BATCH
    batchStart = START_BATCH * batchSize

    while batchStart < len(filings):
        batchEnd = batchStart + batchSize

        print(f'\n-- Batch { batchCounter }')
        build_full_dataset(
            filings = filings.iloc[batchStart:batchEnd],
            X_fname = f'X_{ file_identifier }_{ batchCounter }.npy',
            y_fname = f'y_{ file_identifier }_{ batchCounter }.npy',
            subsample = subsample,
        )

        batchStart += batchSize
        batchCounter += 1

def combine_batched_datasets(file_identifier, num_batches):
    """ Combine saved X_{file_id}, y_{file_id} batch datasets into one .npy file """

    prefs = (f'X_{ file_identifier }', f'y_{ file_identifier }')
    for prefix in prefs:

        print(f'Load { prefix }_*.npy batches')
        datasets = []
        for batch in range(num_batches):
            fname = f'{ prefix }_{ batch }.npy'
            datasets.append(np.load(fname))

        print('Combine batches')
        if len(datasets[0].shape) > 1: # X_full: 2-dimensional so vstack
            fullDataset = np.vstack(datasets)
        else: # y_full: 1-dimensional so concatenate
            fullDataset = np.concatenate(datasets)

        print('Verify rows')
        sumBatchLengths = sum([p.shape[0] for p in datasets])
        assert sumBatchLengths == fullDataset.shape[0], \
            (
                'Sum of batch rows != full dataset rows:' 
                f' { sumBatchLengths } != { fullDataset.shape[0] }'
            )
        
        outFname = prefix + '.npy'
        print(f'Save to { outFname }')
        np.save(outFname, fullDataset)

def save_random_sample(num_rows, src_identifier, dest_identifier, test_start_index=-1):
    """ Save a random sample of the full dataset

        Args:
            num_rows (int): kind of obvious innit
            src_identifier (str):   load from X_{src_identifier}.npy
            dest_identifier (str):  save to X_{dest_identifier}.npy
    """

    print('Load')
    X, y = np.load(f'X_{ src_identifier }.npy'), np.load(f'y_{ src_identifier }.npy')

    if test_start_index > -1:
        print('Subset to training data')
        X, y = X[:test_start_index], y[:test_start_index]

    print('Shuffle')
    np.random.seed(314159)
    shuffledIndex = np.arange(X.shape[0])
    np.random.shuffle(shuffledIndex)
    X = X[shuffledIndex]
    y = y[shuffledIndex]

    print('Save')
    np.save(f'X_{ dest_identifier }.npy', X[:num_rows])
    np.save(f'y_{ dest_identifier }.npy', y[:num_rows])



##############################################
##   Build full dataset from all filings
##############################################

# Used to remove test data from sampled datasets
from test_start_index import test_start_index as TEST_START_INDEX

def search_subarray(sub, full):
    """ Search for subarray in full array 

        Note: this is hacky and has obvious problems and will not work in general,
        but I'm literally using it one time so chill.

        This was used to find the start index of the test portion of the data and
        save it to test_start_index.py, and shall never be used again (Although it 
        was quite fast!). It is preserved here for posterity.
    """

    N = full.shape[0]
    L = sub.shape[0]
    subString = sub.tostring()

    stringSize = 2000000
    stringCount = 0
    stringStart = 0
    found = False
    while stringStart < N:
        print(f'Searching string { stringCount }')
        stringEnd = int(stringStart + stringSize)
        supString = full[stringStart : stringEnd].tostring()
        try:
            i = supString.index(subString)//full.itemsize
        except ValueError:
            print('Not found')
        else:
            print(f'Found match: { i }')
            found = True
            break
        stringStart += stringSize
        stringCount += 1
    
    if not found:
        print('Failed search. Try changing stringSize')
    else:
        startMatch = stringCount*stringSize + i
        print(f'The interior string index was { i } and final stringCount was { stringCount }')
        print(f'Hence, the start index of the match should be { startMatch }')
        print(f'Verify: { np.array_equal(sub, full[startMatch: startMatch + L]) }')
        print(f'First ten elements: \n{ sub[:10] } \n{ full[startMatch : startMatch + 10] }')
        print(f'Ten elements from the middle of the match: '
              f'\n{ sub[int(L/2) : int(L/2) + 10] }'
              f'\n{ full[startMatch + int(L/2) : startMatch + int(L/2) + 10] }')



if __name__ == '__main__':


    ### Steps to build the subsampled dataset
    # build_batched_dataset('subsampled', start_batch=0, subsample=True)
    # combine_batched_datasets('subsampled', num_batches=9)
    # save_random_sample(num_rows=2000000, file_identifier='subsampled_small') ### Not actually used

    pass