"""
    Build a CBOW word2vec model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


#######################################
##   Vocabulary and word freqs
#######################################
with open(os.path.join('D:\\qcf_nlp\\practice_qcf\\analysis', 'dictionary.json'), 'r') as f:
    vocabulary = json.load(f)

inverse_vocabulary = {v: k for k, v in vocabulary.items()}

with open(os.path.join('D:\\qcf_nlp\\practice_qcf\\analysis', 'word_freqs.json'), 'r') as f:
    word_freqs = json.load(f)

def vocab_index(word):
    """ Return index of word in vocabulary """

    try:
        return vocabulary[word]
    except KeyError:
        return vocabulary['<UNK>'] # word not in vocabulary

def index_to_word(i):
    """ Return word corresponding to index in vocabulary """
    return inverse_vocabulary[i]

def cw_to_ci(cw):
    """ Map contextWord to contextIndices 

        cw format: (list(str), str)
        ci format: (list(int), int)
    """
    return (
        [vocab_index(t) for t in cw[0]],    # transform list of context words
        vocab_index(cw[1]),     # transform target word
    )

def ci_to_cw(ci):
    """ Inverse of cw_to_ci """

    return (
        [index_to_word(i) for i in ci[0]],
        index_to_word(ci[1]),
    )


#######################################
##   CBOW_Model class
#######################################
class CBOW_Model(nn.Module):
    """ CBOW word2vec model """

    def __init__(self, V, E, CW):
        """
            Arguments:
                V (int):    vocabulary size
                E (int):    embedding dimension
                CW (int):   context window size (e.g. 5 <=> 2 words before and after target word)
        """

        super(CBOW_Model, self).__init__()
        self.V = V
        self.E = E
        self.CW = CW
        self.embed = nn.Embedding(V, E)
        self.linear = nn.Linear(E, V)

    def forward(self, inputs):
        """
            Arguments:
                inputs (np.Array (N, CW-1)):  indices of context words

            Returns:
                (np.Array (N, V)): scores for words in vocabulary (not softmaxed)
        """

        
        embeds = self.embed(torch.LongTensor(inputs))
        avgEmbeds = torch.mean(embeds, dim=1)
        out = self.linear(avgEmbeds)
        return out


#######################################
##   Trial run
#######################################
def make_sample_CBOW_dataset(fname='93410_0000093410-12-000003_10-Q_output.txt'):
    """ Create a sample dataset from a single document """

    # Open, clean, tokenize an example document
    with open(
            os.path.join('D:\\qcf_nlp\\practice_qcf\\data\\relevant_text', fname), 
            encoding='utf-8'
        ) as f:
        exText = f.read()
    exTokens = clean_tokens(exText)

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
    return contextArr, targetArr


if __name__ == '__main__':

    pass