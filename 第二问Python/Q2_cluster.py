# k-means 聚类
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN,AgglomerativeClustering,Birch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import numpy as np 
import scipy.io as io

# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
# 参考资料 https://zhuanlan.zhihu.com/p/126661239
def minibatchKmeans(X):
    for index, k in enumerate((3,4,5,6)):
        plt.subplot(2,2,index+1)
        y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(X)
        score= metrics.calinski_harabasz_score(X, y_pred)  
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                    transform=plt.gca().transAxes, size=10,
                    horizontalalignment='right')
    plt.show()

def Kmeans3D(data):

    estimator = KMeans(n_clusters=3)  # 构造聚类器
    y = estimator.fit_predict(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=y, marker='*')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='>')
    plt.axis([0, 1, 0, 1])
    plt.show()
def minibatchKmeans(X,k,p=False):
    if p==True:
        for index, kk in enumerate((3,4,5,6)):
            plt.subplot(2,2,index+1)
            y_pred = MiniBatchKMeans(n_clusters=kk).fit_predict(X)
            score= metrics.calinski_harabasz_score(X, y_pred)  
            plt.scatter(X[:, 0], X[:, 1], c=y_pred)
            plt.text(.99, .01, ('k=%d, score: %.2f' % (kk,score)),
                        transform=plt.gca().transAxes, size=10,
                        horizontalalignment='right')
        plt.show()
    else:
        y_pred = KMeans(n_clusters=k).fit_predict(X)
        clusters = unique(y_pred)
        indexdict ={}
        for i in clusters:
            indexarr = np.where(y_pred[:]==i)
            indexdict[i] = indexarr[0]
        return indexdict
def P_Kmeans(X,k,p=False):
    if p==True:
        for index, kk in enumerate((3,4,5,6)):
            plt.subplot(2,2,index+1)
            y_pred = KMeans(n_clusters=kk).fit_predict(X)
            score= metrics.calinski_harabasz_score(X, y_pred)  
            plt.scatter(X[:, 0], X[:, 1], c=y_pred)
            plt.text(.99, .01, ('k=%d, score: %.2f' % (kk,score)),
                        transform=plt.gca().transAxes, size=10,
                        horizontalalignment='right')
        plt.show()
    else:
        y_pred = KMeans(n_clusters=k).fit_predict(X)
        clusters = unique(y_pred)
        indexdict ={}
        for i in clusters:
            indexarr = np.where(y_pred[:]==i)
            indexdict[i] = indexarr[0]
        return indexdict
def Gau(X):
    for index, k in enumerate((3,4,5,6)):
        plt.subplot(2,2,index+1)
        y_pred = GaussianMixture(n_components=k).fit_predict(X)
        score= metrics.calinski_harabasz_score(X, y_pred)  
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                    transform=plt.gca().transAxes, size=10,
                    horizontalalignment='right')
    plt.show()
    y_pred = GaussianMixture(n_components=k).fit_predict(X)
   

def P_DBSCAN(X):
    for index, k in enumerate((3,4,5,6)):
        plt.subplot(2,2,index+1)
        y_pred = DBSCAN(eps=0.30, min_samples=k).fit_predict(X)
        score= metrics.calinski_harabasz_score(X, y_pred)  
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                    transform=plt.gca().transAxes, size=10,
                    horizontalalignment='right')
    plt.show()
def Agg(X):
    # 聚合聚类
    for index, k in enumerate((3,4,5,6)):
        plt.subplot(2,2,index+1)
        y_pred = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        score= metrics.calinski_harabasz_score(X, y_pred)  
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                    transform=plt.gca().transAxes, size=10,
                    horizontalalignment='right')
    plt.show()
def P_Birch(X):
    # BIRCH 聚类（ BIRCH 是平衡迭代减少的缩写，聚类使用层次结构)包括构造一个树状结构，从中提取聚类质心。
    for index, k in enumerate((3,4,5,6)):
        plt.subplot(2,2,index+1)
        y_pred = Birch(threshold=0.01, n_clusters=k).fit_predict(X)
        score= metrics.calinski_harabasz_score(X, y_pred)  
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                    transform=plt.gca().transAxes, size=10,
                    horizontalalignment='right')
    plt.show()
if __name__ =="__main__":
    # 定义数据集
    data_1_predict_raw = pd.read_excel('./data/data_1_knn_IAQI.xlsx',sheet_name='监测点A逐小时污染物浓度与气象实测数据')
    # X_out = data_1_predict_raw[['so2','no2','pm10','pm2.5','o3']]#,'co']]#,'direction']]#,'pressure'
    # X_out = data_1_predict_raw[['so2','no2','pm10','pm2.5','o3',]]#,'IAQI_SO_2','IAQI_NO_2','IAQI_PM_10','IAQI_PM_25','IAQI_O_3']]#,'co']]#,'direction']]#,'pressure'
    # X_out = data_1_predict_raw[['so2','no2','pm10','pm2.5','o3','temperature','humidity','pressure','wind','direction']]
    # X_out = data_1_predict_raw[['temperature','humidity','wind','direction','IAQI_SO_2','IAQI_NO_2','IAQI_PM_10','IAQI_PM_25','IAQI_O_3']]
    # X_out = data_1_predict_raw[['IAQI_SO_2','IAQI_NO_2','IAQI_PM_10','IAQI_PM_25','IAQI_O_3']]
    
    X_out = X_out.values

    scaler = StandardScaler()      
    X = scaler.fit_transform(X_out)  
    # Kmeans3D()
    # minibatchKmeans(X)
    # Gau()
    # Agg(X)
    # P_Birch(X)
    k = 3
    indexdict = P_Kmeans(X,k,True)
    # indexdict = minibatchKmeans(X,k,True)
    X_all = data_1_predict_raw[['so2','no2','pm10','pm2.5','o3','co','temperature','humidity','pressure','wind','direction','aqi']].values
    # 生成mat文件到MATLAB中绘图
    io.savemat('./IAQI_data_3f/X_3all.mat', {'X_all': X_all})
    io.savemat('./IAQI_data_3f/index1.mat', {'index1': indexdict[0]})
    io.savemat('./IAQI_data_3f/index2.mat', {'index2': indexdict[1]})
    io.savemat('./IAQI_data_3f/index3.mat', {'index3': indexdict[2]})
