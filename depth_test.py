import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering

# 305.270752059 rgb/305.270752059.png 305.255445061 depth/305.255445061.png
path = "depth_cut/test4/"
image = cv2.imread(path + '1532.86007349.png')
mask = cv2.imread(path + '1532.873477782.png')
overlay = cv2.addWeighted(image, 1, mask, 0.8, 0)
cv2.imwrite(path + "overlay.png", overlay)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

for num_c, rows in enumerate(gray_mask):
    for num_r, col in enumerate(rows):
        gray_mask[num_c][num_r] = 255 if col > 0 else 0

another = cv2.bitwise_and(gray_image.copy(), gray_mask)

# cv2.imwrite(path + "and.png", another)

# clustering = DBSCAN(min_samples = 3, n_jobs=12)
km = KMeans(n_clusters=3)
# brc = Birch(threshold=5, n_clusters=2)
X = []
for num_c, rows in enumerate(another):
    for num_r, col in enumerate(rows):
        X.append([col,])
# labels = clustering.fit_predict(X)
labels = km.fit_predict(X)
# labels = brc.fit_predict(X)
# labels = OPTICS(min_samples=10).fit_predict(X)


uniq = {}
for a in labels:
    if not (a in uniq):
        uniq[a] = 0
    else:
        uniq[a] +=1
print(uniq)
maxk1 = max(uniq, key=uniq.get)
uniq[maxk1] = 0
maxk2 = max(uniq, key=uniq.get) 

for num_c, rows in enumerate(another):
    for num_r, col in enumerate(rows):
        if (labels[num_c * 1920 + num_r]  == maxk2):
            another[num_c][num_r] = 255
        else:
            another[num_c][num_r] = 0
# print(clustering.labels_)
# cv2.imwrite("305.255445061.png", another)
# gray_mask_not_gray = cv2.cvtColor(another, cv2.COLOR_GRAY2RGB)
# cv2.imwrite("305.255445061_no_gray.png", gray_mask_not_gray)

# another = cv2.addWeighted(gray_mask_not_gray, 1, mask, 0.7, 0)
# cv2.imwrite("305.255445061_mask.png", another)

for num_c, rows in enumerate(another):
    for num_r, col in enumerate(rows):
        if (another[num_c][num_r] == 0):
            mask[num_c][num_r] = (0,0,0)
cv2.imwrite(path + "ready.png", mask)
