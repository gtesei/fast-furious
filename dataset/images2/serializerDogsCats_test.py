import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from glob import glob
from edginess import edginess_sobel

basedir = 'test_dogs_vs_cats'

def features_for(im):
    im = mh.imread(im,as_grey=True).astype(np.uint8)
    #return mh.features.haralick(im).mean(0)
    return np.squeeze(mh.features.haralick(im)).reshape(-1)

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

np.savetxt("test_featuresDogsCatsE.zat", features, delimiter=",")
np.savetxt("test_featuresDogsCats_sobelsE.zat", features_sobels, delimiter=",")










