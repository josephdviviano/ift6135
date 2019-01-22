#
# Advanced neural networks: recurrent neural networks.
# Given at SciNet, 16 October 2017, by Erik Spence.
#
# This file, Learn_Recipes.py, contains code used for lecture 2.  It
# is a script which trains an RNN to predict the next word in a text
# file.  In this case it is used to learn to create cooking recipes.
#

#######################################################################


"""
Learn_Recipes.py contains a script which trains an RNN to predict
the next word in a text file.  This code was originally inspired by
https://github.com/vivshaw/shakespeare-LSTM/blob/master/network/train.py

"""


#######################################################################


# This code was inspired by 
# https://github.com/vivshaw/shakespeare-LSTM/blob/master/network/train.py


#######################################################################


import numpy as np
import os

import keras.models as km
import keras.layers as kl
import shelve


#######################################################################


# Specify some data file names.
datafile0 = 'allrecipes.txt'
datafile = 'recipes.data'
shelvefile = 'recipes.metadata.shelve'
modelfile = 'recipes.model.h5'


#######################################################################


# If the data have already been processed, then don't do it again,
# just read it in.
if (not os.path.isfile('data/' + shelvefile)):

    # Read in the entire file.
    print 'Reading data.'
    f = open(os.path.expanduser(datafile0))
    corpus0 = f.read()
    f.close()

    
    # Separate the punctuation from the words, so that words with
    # punctuation do not get counted as distinct from words without
    # punctuation.  Same for new line characters.
    corpus0 = corpus0.replace(',', ' ,')
    corpus0 = corpus0.replace('(', '( ')
    corpus0 = corpus0.replace(')', ' )')
    corpus0 = corpus0.replace('.', ' .')
    corpus0 = corpus0.replace(';', ' ;')
    corpus0 = corpus0.replace(':', ' :')
    corpus0 = corpus0.replace('!', ' !')
    corpus0 = corpus0.replace('?', ' ?')
    corpus0 = corpus0.replace('\r\n', ' \r\n')


    
    # This is a total hack.  We want to count the multiple-dash
    # 'words' as words.  But below we're going to separate the single
    # dashes from words.  So we do this hack to keep the multiple-dash
    # words intact.  Here we replace the multiple-dash words with
    # dummy strings.
    corpus0 = corpus0.replace('--', 'replaceme1')
    corpus0 = corpus0.replace('--------', 'replaceme2')
    corpus0 = corpus0.replace('------------', 'replaceme3')
    corpus0 = corpus0.replace('--------------------------------', 'replaceme4')
    corpus0 = corpus0.replace('--------------------------------------------------------', 'replaceme5')

    # Separate the dashes from any words they're attached to.
    corpus0 = corpus0.replace('-', ' - ')

    # And here we put the dummy strings back to the original
    # multiple-dash words.
    corpus0 = corpus0.replace('replaceme1', '--')
    corpus0 = corpus0.replace('replaceme2', '--------')
    corpus0 = corpus0.replace('replaceme3', '------------')
    corpus0 = corpus0.replace('replaceme4', '--------------------------------')
    corpus0 = corpus0.replace('replaceme5', '--------------------------------------------------------')


    # Convert the text to lower case.
    corpus0 = corpus0.lower()

    # Split the words by spaces; only take the first 500000 words.
    # This number was chosen based on memory limits and training-time
    # limitations.
    corpus1 = corpus0.split(' ')[0:500000]

    # There are some multiple-new line situations.  We want these to
    # be separated into 2 new line characters.
    corpus2 = [[i[0:2], i[2:]] if i.startswith('\r\n') and len(i) > 2 else [i] for i in corpus1]
    corpus = [i for j in corpus2 for i in j]

    print 'Length of corpus is ', len(corpus)
    
    # Over half the words are spaces.  This is screwing up everything.
    # Remove the spaces and deal with the formatting after the fact.
    # Incidentally, this is extremely slow.  There must be a faster
    # way to do this.
    while(corpus.count('') > 0): corpus.remove('')

    print 'Length of corpus is now ', len(corpus)


    
    # Preprocessing is done.  Now get the unique words, and encode
    # them.
    words = sorted(list(set(corpus)))
    num_words = len(words)
    encoding = {w: i for i, w in enumerate(words)}
    decoding = {i: w for i, w in enumerate(words)}

    print 'We have', num_words, 'different words.'


    
    print 'Processing data.'
    # Chop up the data into x and y, slice into roughly num_chars
    # overlapping 'sentences' of length sentence_length.  Encode the
    # characters.
    sentence_length = 50
    x_data = []
    y_data = []
    for i in range(0, len(corpus) - sentence_length):
        sentence = corpus[i: i + sentence_length]
        next_word = corpus[i + sentence_length]
        x_data.append([encoding[word] for word in sentence])
        y_data.append(encoding[next_word])

    # good word: phronesis
    num_sentences = len(x_data)
    print 'We have', len(x_data), 'sentences.'

    # Create the variables to hold the data as it will be used.
    x = np.zeros((num_sentences, sentence_length, num_words), dtype = np.bool)
    y = np.zeros((num_sentences, num_words), dtype = np.bool)

    # Populate the sentences.
    print 'Encoding data.'
    for i, sentence in enumerate(x_data):
        for t, encoded_word in enumerate(sentence):
            x[i, t, encoded_word] = 1
        y[i, y_data[i]] = 1

        
    # The processing of the data takes a fair amount of time.  Save
    # the data so we don't have to do this again.  We do this in a
    # numpy file since the data is large and the shelve can't handle
    # it.
    print 'Saving processed data.'
    np.save('data/' + datafile + '.x.npy', x)
    np.save('data/' + datafile + '.y.npy', y)

    # Do the same with the metadata.
    print 'Creating metadata shelve file.'
    g = shelve.open('data/' + shelvefile, protocol = 2)
    g['sentence_length'] = sentence_length
    g['num_words'] = num_words
    g['encoding'] = encoding
    g['decoding'] = decoding
    g.close()

    
else:

    # If the data already exists, then use it.
    
    print 'Reading metadata shelve file.'
    g = shelve.open('data/' + shelvefile, flag = 'r', protocol = 2)
    sentence_length = g['sentence_length']
    num_words = g['num_words']
    g.close()

    print 'Reading processed data.'
    x = np.load('data/' + datafile + '.x.npy')
    y = np.load('data/' + datafile + '.y.npy')



# If this is our first rodeo, build the model.
if (not os.path.isfile('data/' + modelfile)):

    print 'Building network.'
    model = km.Sequential()
    model.add(kl.LSTM(256, input_shape = (sentence_length, num_words)))
    model.add(kl.Dense(num_words, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd',
                  metrics = ['accuracy'])

else:

    # Otherwise, use the previously-saved model as our starting point
    # so that we can continue to improve it.
    
    print 'Reading model file.'
    model = km.load_model('data/' + modelfile)


# Fit!  Begin elevator music...
print 'Beginning fit.'
fit = model.fit(x, y, epochs = 200, batch_size = 128, verbose = 2)

# Save the model so that we can use it as a starting point.
model.save('data/' + modelfile)
