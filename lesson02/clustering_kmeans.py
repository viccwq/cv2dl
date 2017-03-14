#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#test code for KMeans
def test_1():
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.cluster import KMeans
    #Generate isotropic Gaussian blobs for clustering.
    X, y = make_blobs(n_samples=300, centers=4,\
                      random_state=0, cluster_std=0.60)
    plt.figure('fig1')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='rainbow');
    plt.show()
    
    est = KMeans(4)  # 4 clusters
    est.fit(X)
    y_kmeans = est.predict(X)
    #c:散点的颜色
    #s：散点的大小
    #cmap : Colormap, optional
    plt.figure("fig2")
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow');
    plt.show()

#test code for digits data
def test_2():
    from sklearn.datasets import load_digits
    from sklearn.cluster import KMeans
    digits = load_digits()
    
    est = KMeans(n_clusters=10)
    clusters = est.fit_predict(digits.data)
    est.cluster_centers_.shape

    fig = plt.figure(figsize=(8, 3))
    for i in range(10):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        ax.imshow(est.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
        
#test code for image
def test_3():      
    from sklearn.datasets import load_sample_image
    from sklearn.cluster import KMeans
    china = load_sample_image("china.jpg")
    plt.figure('china')    
    plt.imshow(china)
    plt.grid(False)
    plt.show()    
    print(china.shape)
    
    X = (china / 255.0).reshape(-1, 3)
    print(X.shape)
    
   # reduce the size of the image for speed
    image = china[::3, ::3]
    n_colors = 64
    
    X = (image / 255.0).reshape(-1, 3)
        
    model = KMeans(n_colors)
    labels = model.fit_predict(X)
    colors = model.cluster_centers_
    new_image = colors[labels].reshape(image.shape)
    new_image = (255 * new_image).astype(np.uint8)
    
    # create and plot the new image
    plt.figure()
    plt.imshow(image)
    plt.title('input')

    plt.figure()
    plt.imshow(new_image)
    plt.title('{0} colors'.format(n_colors))

    
if __name__ == '__main__':
    
    print("Welcome to My blog!")
    #test_1()
    #test_2()
    test_3()
    

