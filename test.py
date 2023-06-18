from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import kmeans
from sklearn.datasets import load_digits

# digits = load_digits()
# for i in range(10):
#     label = digits.target[i]
#     img = digits.images[i]
#     img_cluster = np.copy(img)
#     center, labels = kmeans.kMeans(img,5)
#     for i in range(len(img)):
#         img_cluster[i] = center[int(labels[i])]
#     fig, axs = plt.subplots(1, 2)
#     axs[0].imshow(img)
#     axs[0].set_title("Hình gốc")
#     axs[1].imshow(img_cluster)
#     axs[1].set_title('Sau khi xử lý')
#     plt.show()

img = cv2.imread(r"D:\start\image\girl3.jpg")
print(img.shape)
X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
re = np.zeros_like(X)
centers, labels = kmeans.kMeans(X,3)

for i in range(len(X)):
    re[i] = centers[int(labels[i])]
re = re.reshape((img.shape[0], img.shape[1], img.shape[2]))
fig, axs = plt.subplots(1, 2)
axs[0].matshow(img)
axs[0].set_title("Hình gốc")
axs[1].imshow(re)
axs[1].set_title('Sau khi xử lý')
plt.show()
