#
# Advanced neural networks: recurrent neural networks.
# Given at SciNet, 16 October 2017, by Erik Spence.
#
# This file, Generate_Recipes.py, contains code used for lecture 2.
# It is a script which takes a trained RNN and uses it to generate
# text.  In this case it is used to generate cooking recipes.
#

#######################################################################


"""
Generate_Recipes.py contains a script which uses a trained RNN to
generate text.  This code was inspired by
https://github.com/vivshaw/shakespeare-LSTM/blob/master/network/generate.py

"""

#######################################################################


# This code was inspired by 
# https://github.com/vivshaw/shakespeare-LSTM/blob/master/network/generate.py


#######################################################################


import numpy as np

import keras.models as km
import shelve
import random


#######################################################################


# Specify the model and metadata files.
shelvefile = 'recipes.metadata.shelve'
modelfile = 'recipes.model.h5'


#######################################################################


# Get the model.
print 'Reading model file.'
model = km.load_model('data/' + modelfile)

# Get the meta-data.
print 'Reading metadata shelve file.'
g = shelve.open('data/' + shelvefile, protocol = 2, flag = 'r')
sentence_length = g['sentence_length']
num_words = g['num_words']
encoding = g['encoding']
decoding = g['decoding']
g.close()


# Randomly choose 50 words from the dictionary of words as our
# starting sentence.
seed = []
for i in range(sentence_length):
    seed.append(decoding[random.randint(0, num_words - 1)])


# Encode the seed sentence.
x = np.zeros((1, sentence_length, num_words), dtype = np.bool)
for i, w in enumerate(seed):
    x[0, i, encoding[w]] = 1

text = ''

# Run the seed sentence through the model.  Add the output to the
# generated text.  Take the output and append it to the seed sentence
# and remove the first word from the seed sentence.  Then repeat until
# you've generated as many words as you like.
for i in range(400):

    # Get the most-probably next word.
    pred = np.argmax(model.predict(x, verbose = 0))

    # Add it to the generated text.
    text += decoding[pred] + ' '

    # Encode the next word.
    next_word = np.zeros((1, 1, num_words), dtype = np.bool)
    next_word[0, 0, pred] = 1

    # Concatenate the next word to the seed sentence, but leave off
    # the first element so that the length stays the same.
    x = np.concatenate((x[:, 1:, :], next_word), axis = 1)

    
# Print out the generated text.
print "Our recipe:"
print text
