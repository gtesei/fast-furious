####################################################
# Prepare small collection 

import os
import shutil
import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from glob import glob
from edginess import edginess_sobel



def features_for(im):
    im = mh.imread(im,as_grey=True).astype(np.uint8)
    #return mh.features.haralick(im).mean(0)
    return np.squeeze(mh.features.haralick(im)).reshape(-1)


os.mkdir("small_collection") ## 3000 train , 3000 test 

## train, cats 
basedir = 'train-dogs-vs-cats'
images = glob('{}/*.jpg'.format(basedir))
ims = len(images)
for i in range(1,1501):
    im = 'train-dogs-vs-cats/cat.'+str(i)+'.jpg'
    dst = 'small_collection/cat.'+str(i)+'.jpg'
    print('picking ' +str(im) +' ...')
    shutil.copy(im, dst)

    
## train, dogs 
basedir = 'train-dogs-vs-cats'
images = glob('{}/*.jpg'.format(basedir))
ims = len(images)
for i in range(1,1501):
    im = 'train-dogs-vs-cats/dog.'+str(i)+'.jpg'
    dst = 'small_collection/dog.'+str(i)+'.jpg'
    print('picking ' +str(im) +' ...')
    shutil.copy(im, dst)



## test
basedir = 'test_dogs_vs_cats'
images = glob('{}/*.jpg'.format(basedir))
for i in range(1,3001):
    im = 'test_dogs_vs_cats/'+str(i)+'.jpg'
    dst = 'small_collection/'+str(i)+'.jpg'
    print('picking ' +str(im) +' ...')
    shutil.copy(im, dst)


####################################################
# Xtrain1, Xtrain2, Ytrain

basedir = 'train-dogs-vs-cats'


features = []
sobels = []
labels = []
labelVect = [] # cat 0 , dog 1 
images = glob('{}/*.jpg'.format(basedir))
for im in images:
    features.append(features_for(im))
    sobels.append(edginess_sobel(mh.imread(im, as_grey=True)))
    #labels.append(im[:-len('00.jpg')])
    if ("train-dogs-vs-cats/cat" in im[:-len('.jpg')]):
        labels.append('cat')  
        labelVect.append(np.array([0]))
    elif ("train-dogs-vs-cats/dog" in im[:-len('.jpg')]):
        labels.append('dog')
        labelVect.append(np.array([1]))
    else: 
        raise Exception ("unrecognized label:"+str(im[:-len('.jpg')]))


features = np.array(features)
labels = np.array(labels)
n = features.shape
nl = labels.shape
print('features='+str(n))
print(str(features))
print ('labels='+str(nl))
print(str(labels))

features_sobels = np.hstack([np.atleast_2d(sobels).T,features])

np.savetxt("Xtrain1.zat", features, delimiter=",")
np.savetxt("Xtrain2.zat", features_sobels, delimiter=",")
np.savetxt("Ytrain.zat", labelVect, delimiter=",")

scores = cross_validation.cross_val_score(LogisticRegression(), features, labels, cv=5)
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features]: {}%'.format(0.1* round(1000*scores.mean())))

scores = cross_validation.cross_val_score(LogisticRegression(), np.hstack([np.atleast_2d(sobels).T,features]), labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features + sobel]: {}%'.format(0.1* round(1000*scores.mean())))


####################################################
# Xtest1, Xtest2

basedir = 'test_dogs_vs_cats'

features = []
sobels = []
images = glob('{}/*.jpg'.format(basedir))
ims = len(images)
for i in range(1,ims+1):
    im = 'test_dogs_vs_cats/'+str(i)+'.jpg'
    print('processing ' +str(im) +' ...')
    #for im in images:
    features.append(features_for(im))
    sobels.append(edginess_sobel(mh.imread(im, as_grey=True)))


features = np.array(features)
n = features.shape
print('features='+str(n))
print(str(features))

features_sobels = np.hstack([np.atleast_2d(sobels).T,features])

np.savetxt("Xtest1.zat", features, delimiter=",")
np.savetxt("Xtest2.zat", features_sobels, delimiter=",")

####################################################
# Xtrain3 

print('SURFing ...')

from sklearn.cluster import KMeans
from mahotas.features import surf


basedir = 'small_collection'
images = glob('{}/*.jpg'.format(basedir))
alldescriptors = []
i = 0;
for im in images:
    im = mh.imread(im, as_grey=1)
    im = im.astype(np.uint8)
    alldescriptors.append(surf.surf(im, descriptor_only=True))
    i += 1
    print ('image:'+str(i))


print('Descriptors done')
k = 256
km = KMeans(k)

concatenated = np.concatenate(alldescriptors)
#concatenated = concatenated[::64]
concatenated = concatenated[:64]
print('k-meaning...')
km.fit(concatenated)
features = []


basedir = 'train-dogs-vs-cats'
images = glob('{}/*.jpg'.format(basedir))
for im in images:
    im = mh.imread(im, as_grey=1)
    im = im.astype(np.uint8)
    d = surf.surf(im, descriptor_only=True)
    c = km.predict(d)
    features.append(
                    np.array([np.sum(c == i) for i in xrange(k)])
                    )

features = np.array(features)

np.savetxt("Xtrain3.zat", features, delimiter=",")
np.savetxt("SURF_concatenated.zat", concatenated, delimiter=",")

print('predicting...')
scoreSURFlr = cross_validation.cross_val_score(LogisticRegression(), features, labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Log. Reg [SURF features]: %s%%' % (0.1* round(1000*scoreSURFlr.mean())))


####################################################
# Xtest3 

features = []
basedir = 'test_dogs_vs_cats'
images = glob('{}/*.jpg'.format(basedir))
ims = len(images)
    #for im in images:
for i in range(1,ims+1):
    im = 'test_dogs_vs_cats/'+str(i)+'.jpg'
    print('processing ' +str(im) +' ...')
    im = mh.imread(im, as_grey=1)
    im = im.astype(np.uint8)
    d = surf.surf(im, descriptor_only=True)
    c = km.predict(d)
    features.append(
                    np.array([np.sum(c == i) for i in xrange(k)])
                    )


features = np.array(features)

np.savetxt("Xtest3.zat", features, delimiter=",")

