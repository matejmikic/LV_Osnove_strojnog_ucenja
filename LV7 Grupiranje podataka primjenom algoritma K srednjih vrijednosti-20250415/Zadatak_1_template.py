import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
#X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
#plt.figure()
#plt.scatter(X[:,0],X[:,1])
#plt.xlabel('$x_1$')
#plt.ylabel('$x_2$')
#plt.title('podatkovni primjeri')
#plt.show()


#km = KMeans(n_clusters=4, init='k-means++', n_init=5, random_state=0)



def prikazi_grupirane_podatke(n_samples, flag_c,n_clusters):
    # generiranje podatkovnih primjera
    X = generate_data(n_samples, flag_c)
    km = KMeans(n_clusters, init='k-means++', n_init=5, random_state=0)
    km.fit(X)
    labels = km.predict(X)

    # prikazi primjere u obliku dijagrama rasprsenja
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("podatkovni primjeri")
    plt.show()


for i in range(1,6):
    for j in range(1,6):
        prikazi_grupirane_podatke(500,i,j)