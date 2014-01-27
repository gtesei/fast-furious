import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from glob import glob
from edginess import edginess_sobel

#basedir = 'simple-dataset'

basedir = 'train-dogs-vs-cats'

def features_for(im):
    im = mh.imread(im,as_grey=True).astype(np.uint8)
    #return mh.features.haralick(im).mean(0)
    return np.squeeze(mh.features.haralick(im)).reshape(-1)

features = []
sobels = []
labels = []
labelVect = [] # cat 1 , dog 0 
images = glob('{}/*.jpg'.format(basedir))
for im in images:
    features.append(features_for(im))
    sobels.append(edginess_sobel(mh.imread(im, as_grey=True)))
    #labels.append(im[:-len('00.jpg')])
    if ("train-dogs-vs-cats/cat" in im[:-len('.jpg')]):
        labels.append('cat')  
        labelVect.append(np.array([1]))
    elif ("train-dogs-vs-cats/dog" in im[:-len('.jpg')]):
        labels.append('dog')
        labelVect.append(np.array([0]))
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

np.savetxt("featuresDogsCatsE.zat", features, delimiter=",")
np.savetxt("featuresDogsCats_sobelsE.zat", features_sobels, delimiter=",")
#np.savetxt("labelsDogsCatsE.zat", labelVect, delimiter=",")

scores = cross_validation.cross_val_score(LogisticRegression(), features, labels, cv=5)
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features]: {}%'.format(0.1* round(1000*scores.mean())))

scores = cross_validation.cross_val_score(LogisticRegression(), np.hstack([np.atleast_2d(sobels).T,features]), labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features + sobel]: {}%'.format(0.1* round(1000*scores.mean())))

####################################################
print('SURFing ...')
tfeatures = features
from sklearn.cluster import KMeans
from mahotas.features import surf

alldescriptors = []
for im in images:
    im = mh.imread(im, as_grey=1)
    im = im.astype(np.uint8)
    alldescriptors.append(surf.surf(im, descriptor_only=True))

print('Descriptors done')
k = 256
km = KMeans(k)

concatenated = np.concatenate(alldescriptors)
concatenated = concatenated[::64]
print('k-meaning...')
km.fit(concatenated)
features = []

for d in alldescriptors:
    c = km.predict(d)
    features.append(
                    np.array([np.sum(c == i) for i in xrange(k)])
                    )
features = np.array(features)

np.savetxt("featuresDogsCatsSURF.zat", features, delimiter=",")

print('predicting...')
scoreSURFlr = cross_validation.cross_val_score(LogisticRegression(), features, labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Log. Reg [SURF features]: %s%%' % (0.1* round(1000*scoreSURFlr.mean())))

print('combined...')
allfeatures = np.hstack([features, tfeatures])
scoreSURFplr = cross_validation.cross_val_score(LogisticRegression(), allfeatures, labels, cv=5).mean()

print('Accuracy (5 fold x-val) with Log. Reg [All features]: %s%%' % (0.1* round(1000*scoreSURFplr.mean())))







