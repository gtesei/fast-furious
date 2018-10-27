import os
from shutil import copyfile

with open('./sample_id.txt') as f:
    ids = f.readlines()

ids = [str(_id)[:-1]+'.png' for _id in ids]

for _id in ids:
    copyfile('./data/train/images/'+str(_id),'./data_sample/train/images/'+str(_id))
    copyfile('./data/train/masks/'+str(_id),'./data_sample/train/masks/'+str(_id))
    


