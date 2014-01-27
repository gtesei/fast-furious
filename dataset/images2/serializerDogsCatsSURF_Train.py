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

features = []
sobels = []
labels = []

####################################################
print('SURFing ...')
tfeatures = features
from sklearn.cluster import KMeans
from mahotas.features import surf


basedir = 'small_train-dogs-cats'
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
concatenated = concatenated[::64]
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
    if ("train-dogs-vs-cats/cat" in im[:-len('.jpg')]):
        labels.append('cat')
    elif ("train-dogs-vs-cats/dog" in im[:-len('.jpg')]):
        labels.append('dog')
    else:
        raise Exception ("unrecognized label:"+str(im[:-len('.jpg')]))

features = np.array(features)

np.savetxt("featuresDogsCatsSURF.zat", features, delimiter=",")
np.savetxt("SURF_concatenated.zat", concatenated, delimiter=",")

print('predicting...')
scoreSURFlr = cross_validation.cross_val_score(LogisticRegression(), features, labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Log. Reg [SURF features]: %s%%' % (0.1* round(1000*scoreSURFlr.mean())))







