#
# Advanced neural networks: generative adversarial networks.
# Given at SciNet, 30 October 2017, by Erik Spence.
#
# This file, MNIST_wgan.py, contains code used for lecture 3.  It is a
# script which trains a generative adversarial network on the MNIST
# dataset.
#

#######################################################################


"""
MNIST_wgan.py contains a script which trains a generative
adversarial network on the MNIST dataset.  This code was inspired by
many online examples, but the main one was
https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html

"""


#######################################################################


# This code was inspired by many many online examples, but the main
# one was
# https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html


#######################################################################


import os

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import keras.backend as K
from keras.datasets import mnist
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras.initializers as ki


############################################3

# Size of our noise input.
Z_SIZE = 100

# number of iterations D is trained per G iteration.
D_ITERS = 5

# SGD batch size and number of total training iterations.
BATCH_SIZE = 100
ITERATIONS = 25000


############################################3


# Sometimes it's helpful to specify that you don't want the very first
# available GPU on a device.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Probably unnecessary.
K.set_image_dim_ordering('tf')


############################################3


def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)


############################################3


def create_D():

    """
    This function creates the discriminator network for the WGAN.

    Inputs: None.

    Outputs:

    - Keras NN model, which accepts a (28, 28, 1) input and has two
      outputs, a single digit indicating whether the model is real and
      a 10 digit vector indictating the digit.

    """

    
    # If the model has already been defined we can skip this whole
    # section.
    if (not os.path.isfile('data/wgan.discriminator.h5')):

        # Because the weights are clipped at +/- 0.01 it's best to
        # initialize the weights with a tighter-than-usual normal
        # distribution.
        weight_init = ki.RandomNormal(mean = 0.0, stddev = 0.02)

        # The input layer.
        input_image = kl.Input(shape=(28, 28, 1))

        # Add a convolution layer and dropout, several times.
        x = kl.Conv2D(64, (5, 5), strides = (2,2), padding = 'same', 
                      kernel_initializer = weight_init)(input_image)
        x = kl.LeakyReLU(alpha = 0.2)(x)
        x = kl.Dropout(0.4)(x)
        
        x = kl.Conv2D(128, (5, 5), strides = (2,2), padding='same',
                      kernel_initializer = weight_init)(x)
        x = kl.LeakyReLU(alpha = 0.2)(x)
        x = kl.Dropout(0.4)(x)
        
        x = kl.Conv2D(256, (5, 5), strides = (2,2), padding='same',
                      kernel_initializer = weight_init)(x)
        x = kl.LeakyReLU(alpha = 0.2)(x)
        x = kl.Dropout(0.4)(x)

        x = kl.Conv2D(512, (5, 5), strides = (1,1), padding='same',
                      kernel_initializer = weight_init)(x)
        x = kl.LeakyReLU(alpha = 0.2)(x)
        x = kl.Dropout(0.4)(x)

        # Flatten the last layer.
        features = kl.Flatten()(x)

        # Specify the two output variables: whether the input was an
        # image from the real dataset, and which digit it was.
        output_status = kl.Dense(1, activation = 'linear')(features)
        output_class = kl.Dense(10, activation = 'softmax')(features)

        # Create the actual model.
        model = km.Model(inputs = input_image, name = 'D',
                         outputs = [output_status, output_class])

        # We need to compile it now, in case the model is reloaded
        # from the file later.
        model.compile(optimizer = ko.RMSprop(lr = 0.00005),
                      loss = [wasserstein, 'sparse_categorical_crossentropy'])
        
    else:

        # Otherwise just read the existing model from file.
        print "Reading existing discriminator model."
        model = km.load_model('data/wgan.discriminator.h5', \
                              custom_objects = {'wasserstein': wasserstein})


    return model


#############################################


def create_G():

    """
    This function creates the generator network for the WGAN.

    Inputs: None.

    Outputs:

    - Keras NN model, which accepts an 100-element vector of noise,
      and a requested digit as input, and outputs a (28, 28, 1) image.

    """

    # If the model has already been defined we can skip this whole
    # section.
    if (not os.path.isfile('data/wgan.generator.h5')):

        # The size of our dictionary.
        DICT_LEN = 10

        # The dimension of our embedding space.
        EMBEDDING_LEN = Z_SIZE
        
        # Because the weights are clipped at +/- 0.01 it's best to
        # initialize the weights with a tighter-than-usual normal
        # distribution.
        weight_init = ki.RandomNormal(mean = 0.0, stddev = 0.02)

        # This is the digit that we are requesting the network to
        # generate.
        input_class = kl.Input(shape=(1, ), dtype='int32')

        # The embedding layer.
        e = kl.Embedding(DICT_LEN, EMBEDDING_LEN,
                         embeddings_initializer = 'glorot_uniform')(input_class)

        # Because we're just going straight to element-by-element
        # multiplication with the input noise, we need to flatten the
        # embedding layer.
        embedded_class = kl.Flatten()(e)

        # The noise input.
        input_z = kl.Input(shape = (Z_SIZE, ))

        # Element-by-element multiplication (hadamard product) between
        # the embedding layer and the input noise.
        h = kl.multiply([input_z, embedded_class], name='h')

        # The image starting dimension.  This just makes keeping the
        # dimensions straight easier.
        dim = 7

        # Start by adding a fully-connected layer, then reshape into a
        # set of squares and not-really feature maps.
        x = kl.Dense(256 * 7 * 7)(h)
        x = kl.LeakyReLU()(x)
        x = kl.Reshape((7, 7, 256))(x)

        # Now upsample, deconvolve.  Do this twice.
        x = kl.UpSampling2D(size = (2, 2))(x)
        x = kl.Conv2DTranspose(128, (5, 5), padding = 'same',
                      kernel_initializer = weight_init)(x)
        x = kl.LeakyReLU()(x)

        x = kl.UpSampling2D(size = (2, 2))(x)
        x = kl.Conv2DTranspose(64, (5, 5), padding = 'same',
                      kernel_initializer = weight_init)(x)
        x = kl.LeakyReLU()(x)

        # Now add two more deconvolution layers, reducing the number
        # of feature maps as we go.
        x = kl.Conv2DTranspose(32, (5, 5), padding = 'same',
                      kernel_initializer = weight_init)(x)
        x = kl.LeakyReLU()(x)
        
        x = kl.Conv2DTranspose(1, (5, 5),padding = 'same',
                      kernel_initializer = weight_init,
                      activation = 'tanh')(x)

        # Create the model.
        model = km.Model(inputs=[input_z, input_class], outputs=x)

    else:

        # Otherwise just read the model from the file.
        print "Reading existing generator model."
        model = km.load_model('data/wgan.generator.h5')

    return model


####################################################################

                    
def plot_images(generator, noise, classes, step = 0):

    """
    This function creates a 4 x 4 grid of image plots, based on the
    input noise and the desired clases.
    
    Inputs: 

    - generator: NN generator model, used to generate the images.
   
    - noise: a (16, 100) matrix of input noise.
    
    - classes: a 16-element vector if input digits, 0-9.

    - step: the step index, added to the output file name.

    Outputs: None.

    """

    # The filename into which the image is saved.
    filename = "data/mnist_%d.png" % step

    # Create the images.
    images = generator.predict([noise, classes])

    # Create an empty figure.
    plt.figure(figsize = (10, 10))

    # For each image...
    for i in range(images.shape[0]):

        # Change to the next subplot.
        plt.subplot(4, 4, i + 1)

        # Rescale the image back to its 0-255 form.
        image = images[i, :, :, :] * 127.5 + 127.5

        # Reshape to remove the extra dimensions.
        image = np.reshape(image, [28, 28])

        # Plot the image.
        plt.imshow(image, cmap = 'gray_r')

        # Plot the digit that that particular image should be.
        plt.text(23, 26, str(classes[i][0]), fontsize = 30, color = 'r')

        # Turn off the axis.
        plt.axis('off')

    # Tighten up the boundaries, and save.
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
    
            
#############################################


# Now we begin the actual script which creates the networks and does
# the training.


# Create the input layers.
input_z = kl.Input(shape = (Z_SIZE, ))
input_class = kl.Input(shape = (1,), dtype = 'int32')

# Create the networks.
D = create_D()
G = create_G()

# create combined D(G) model.
output_status, output_class = D(G(inputs = [input_z, input_class]))
DG = km.Model(inputs = [input_z, input_class],
              outputs = [output_status, output_class])

# Turn off D before compiling.
DG.get_layer('D').trainable = False

# Compile the generator, through the compined model.
DG.compile(optimizer = ko.RMSprop(lr = 0.00005),
           loss = [wasserstein, 'sparse_categorical_crossentropy'])


#############################################

# But we don't have any data yet!

# load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# use all available 70k samples
X_train = np.concatenate((X_train, X_test))
y_train = np.concatenate((y_train, y_test))

# convert to -1..1 range because we're using tanh as an output.
# Reshape to (sample_i, 28, 28, 1)
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis = 3)


#############################################

# Define the variables which will hold the fitting losses.
DG_losses = []
D_true_losses = []
D_fake_losses = []

# Create the noise data and classes we will use for the plotting.
noise_input = npr.normal(0., 1.0, size = [16, 100])
noise_classes = npr.randint(0, 10, 16).reshape(-1, 1)

# Let us iterate!
for it in range(ITERATIONS):

    # The discriminator is trained many times for each training of the
    # generator.  How many times?  This many times.
    if (it % 1000) < 25 or it % 500 == 0: # 25 times in 1000, every 500th
        d_iters = 100
    else:
        d_iters = D_ITERS

    # Now iterate over the training of the discriminator.
    for d_it in range(d_iters):

        # unfreeze D
        D.trainable = True
        for layer in D.layers: layer.trainable = True

        # Clip the discriminator weights.
        for layer in D.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            layer.set_weights(weights)


        # Draw random samples from the real images.
        index = npr.choice(len(X_train), BATCH_SIZE,
                           replace = False)
        real_images = X_train[index]
        real_images_classes = y_train[index]

        # And do the training on this batch of real images.  Note that
        # we've chosen "y = -1" to correspond to "real" images.
        D_loss = D.train_on_batch(real_images, [-np.ones(BATCH_SIZE),
                                                real_images_classes])

        # Save the loss.
        D_true_losses.append(D_loss)

        # Now create the fake images, based on the current state of
        # the generator.
        zz = npr.normal(0., 1., (BATCH_SIZE, Z_SIZE))
        generated_classes = npr.randint(0, 10, BATCH_SIZE).reshape(-1,1)
        generated_images = G.predict([zz, generated_classes])

        # Now do the dicriminator training on the fake images.  Again,
        # note that "y = 1" corresponds to "fake" images.
        D_loss = D.train_on_batch(generated_images, [np.ones(BATCH_SIZE),
                                                     generated_classes])
        D_fake_losses.append(D_loss)


    # We're done training the discriminator, for this iteration.  Now
    # train the generator.

    # Freeze D.
    D.trainable = False
    for layer in D.layers: layer.trainable = False
        
    # Create some noise, and fake classes.
    zz = npr.normal(0., 1., (BATCH_SIZE, Z_SIZE)) 
    generated_classes = npr.randint(0, 10, BATCH_SIZE).reshape(-1, 1)

    # Now train the combined DG on the generated data, but with only
    # training G.
    DG_loss = DG.train_on_batch(
        [zz, generated_classes],
        [-np.ones(BATCH_SIZE), generated_classes])

    # Again, save the losses.
    DG_losses.append(DG_loss)

    # Every 50 iterations, print out the losses.
    if (it % 50 == 0):
        log_mesg = "%d: [D true loss: %f]" % (it, D_true_losses[-1][0])
        log_mesg = "%s  [D fake loss: %f]" % (log_mesg, D_fake_losses[-1][0])
        log_mesg = "%s  [DG loss: %f]" % (log_mesg, DG_losses[-1][0])
        log_mesg = "%s  [Total loss: %f]" % (log_mesg,
                                             D_true_losses[-1][0] +
                                             D_fake_losses[-1][0])
        print(log_mesg)

    # Every 1000 iterations, save the models.
    if (it % 1000 == 0):
        D.save('data/wgan.discriminator.h5')
        G.save('data/wgan.generator.h5')

    # Every 10 iterations, save the image, so we can create an animation.
    if (it % 10 == 0):
        plot_images(G, noise_input, noise_classes, step = it)
