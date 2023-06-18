import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)


def find_index(arr,value):
    return np.where(arr == value)[0]

def norm_2(x, center):
    return np.sum((x - center)**2)

def initialization(X,k): # hàm khởi tạo
    return X[np.random.choice(X.shape[0], k, replace=False)]

# def assign_labels(X,M):
#     labels = np.array([])
#     for x_i in X:
#         dis_manage = np.array([])
#         for center in M:
#             dis_manage = np.append(dis_manage,norm_2(x_i,center))
#         labels = np.append(labels,np.argmin(dis_manage))

#     return labels
        
def assign_labels(X,M):
    labels = np.array([])
    for x_i in X:
        dis_manage = np.array([])
        for center in M:
            dis_manage = np.append(dis_manage,norm_2(x_i,center))

        if len(dis_manage) > 0:
            labels = np.append(labels,np.argmin(dis_manage))
        else:
            labels = np.append(labels, -1) # gán một nhóm khác
    return labels

# def update_center(X,labels,K):
#     M = np.zeros((K,X.shape[1]))
#     for i in range(K):
#         indies = find_index(labels,i)
#         for indi in indies:
#             M[i] += X[indi]
#         M[i] /= len(indies)
#     return M

def update_center(X,labels,K):
    M = np.zeros((K,X.shape[1]))
    for i in range(K):
        indies = find_index(labels,i)
        if len(indies) > 0:
            for indi in indies:
                M[i] += X[indi]
            M[i] /= len(indies)
        else:
            # trả lại giá trị không thay đổi nếu không có phần tử trong nhóm
            M[i] = M[i]
    return M


def stop(centers, new_centers, tol=1e-5):
    return np.allclose(centers, new_centers, atol=tol)



# def stop(centers, new_centers):
#     return np.allclose(centers, new_centers, rtol=1e-5, atol=1e-8)
# def stop(centers, new_centers):
#     return np.allclose(np.round(centers, decimals=5), np.round(new_centers, decimals=5))




def kMeans(X,K):
    M = initialization(X,K)
    centers_manage = [M]
    count = 1
    while True:
        labels = assign_labels(X,M)
        M = update_center(X,labels,K)
        print(M)
        centers_manage.append(M)
        if stop(M,centers_manage[count - 1]):
            break
        count += 1

    return centers_manage[-1], labels


def display(X,M,labels):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=labels)
    ax.scatter(M[:,0], M[:,1], marker='x', s=200, linewidths=3, color='r')
    ax.set_title('K-means clustering results')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()



