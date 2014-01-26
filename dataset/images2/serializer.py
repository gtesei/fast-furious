import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from glob import glob
from edginess import edginess_sobel

basedir = 'simple-dataset'

#basedir = 'train-dogs-vs-cat'

def features_for(im):
    im = mh.imread(im,as_grey=True).astype(np.uint8)
    return mh.features.haralick(im).mean(0)

features = []
sobels = []
labels = []
labelVect = [] # building [1 0 0] , scene [0 1 0] , text [0 0 1]
images = glob('{}/*.jpg'.format(basedir))
for im in images:
    features.append(features_for(im))
    sobels.append(edginess_sobel(mh.imread(im, as_grey=True)))
    labels.append(im[:-len('00.jpg')])
    if ("building" in im[:-len('00.jpg')]):  
        labelVect.append(np.array([1,0,0]))
    elif ("scene" in im[:-len('00.jpg')]):
        labelVect.append(np.array([0,1,0]))
    elif ("text" in im[:-len('00.jpg')]):
        labelVect.append(np.array([0,0,1]))    
    else: 
        raise Exception ("unrecognized label:"+str(im[:-len('00.jpg')]))
    


features = np.array(features)
labels = np.array(labels)

n = features.shape;
nl = labels.shape;

print('features='+str(n))
print(str(features))
print ('labels='+str(nl))
print(str(labels))

np.savetxt("features.zat", features, delimiter=",")
np.savetxt("labels.zat", labelVect, delimiter=",")

scores = cross_validation.cross_val_score(LogisticRegression(), features, labels, cv=5)
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features]: {}%'.format(0.1* round(1000*scores.mean())))

scores = cross_validation.cross_val_score(LogisticRegression(), np.hstack([np.atleast_2d(sobels).T,features]), labels, cv=5).mean()
print('Accuracy (5 fold x-val) with Logistic Regrssion [std features + sobel]: {}%'.format(0.1* round(1000*scores.mean())))

