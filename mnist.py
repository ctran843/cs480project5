
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014
import pylab as pl
import numpy as np
import mlp
import _pickle, gzip

from matplotlib import pyplot as plt

# Read the dataset in (code from sheet)
f = gzip.open('mnist.pkl.gz', 'rb')
tset, vset, teset = _pickle.load(f, encoding='latin1')
f.close()
#tset[0][i] is the image
#tset[1][i] is the label

#create two data sets of images of 7 and 9 from original data set
seven_image_set = []
nine_image_set = []
for i in range(len(tset[0])):
    if (tset[1][i] == 7):
        seven_image_set.append(tset[0][i])
    elif (tset[1][i] == 9):
        nine_image_set.append(tset[0][i])
print(seven_image_set[0][0])
nread = 200
# Just use the first few images
#training images
train_in = tset[0][:nread, :]


# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
#training labels
train_tgt = np.zeros((nread, 10))
for i in range(nread):
    train_tgt[i, tset[1][i]] = 1

#test images
test_in = teset[0][:nread, :]

#test labels
test_tgt = np.zeros((nread, 10))
for i in range(nread):
    test_tgt[i, teset[1][i]] = 1

# We will need the validation set
valid_in = vset[0][:nread, :]

valid_tgt = np.zeros((nread, 10))
for i in range(nread):
    valid_tgt[i, vset[1][i]] = 1

first_image = nine_image_set[0]
#first_image = train_in[0]
#first_image = test_in[0]
#first_image = valid_in[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

for i in [1,2,5,10,20]:
    print("----- "+str(i))
    net = mlp.mlp(train_in, train_tgt, i,outtype='softmax')
    net.earlystopping(train_in, train_tgt, valid_in, valid_tgt, 0.1)
    net.confmat(test_in, test_tgt)