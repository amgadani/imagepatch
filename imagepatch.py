from sklearn.datasets import load_sample_image
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.neighbors import LSHForest
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import datetime

k = 5

img = misc.imread('scales.png')
data = extract_patches_2d(img, (k,k))
print('Extracted patches: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
data = data.reshape(-1, data.shape[0]).transpose()
print('Reshaped patches: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

lshf = LSHForest(random_state=42)
lshf.fit(data)
print('Fit lshf: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

# X_test = np.random.randint(255, size=k*k*3).reshape(1,-1)
# distances, indices = lshf.kneighbors(X_test, n_neighbors=2)
# print indices
# print distances
# print data[indices]
# import code
# code.interact(local=locals())
##how do i handle edges??
img_w = 100
img_h = 100
result =  np.zeros((img_w,img_h,3), np.uint8)
for i in range(img_w):
    for j in range(0,k/2):
        result[i][j] = img[np.random.randint(img.shape[0])][np.random.randint(img.shape[1])]
        result[j][i] = img[np.random.randint(img.shape[0])][np.random.randint(img.shape[1])]


#fill in
for i in range(k/2,img_w-k/2):
    print('row ' + str(i) + ': {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    for j in range(k/2, img_h-k/2):
        neighborhood = result[(i-k/2):(i+k/2+1), (j-k/2):(j+k/2+1)]
        neighborhood = neighborhood.reshape(1,-1)
        distances, indices = lshf.kneighbors(neighborhood, n_neighbors=1)
        result[i][j] = data[indices[0][0]][(k/2*k/2*3):((k/2*k/2+1)*3)]
        # print indices
        # print distances
        # print data[indices]

#show
plt.imshow(result)
plt.show()
