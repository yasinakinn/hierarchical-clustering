import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dataset = pd.read_csv('customers_dataset.csv')

X = dataset.iloc[:,2:8].values


ward = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
complete = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
average = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')
single = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')

ward_predict = ward.fit_predict(X)
complete_predict = complete.fit_predict(X)
average_predict = average.fit_predict(X)
single_predict = single.fit_predict(X)

plt.scatter(X[ward_predict==0,0],X[ward_predict==0,1],s=100, c='red')
plt.scatter(X[ward_predict==1,0],X[ward_predict==1,1],s=100, c='blue')
plt.scatter(X[ward_predict==2,0],X[ward_predict==2,1],s=100, c='green')
plt.scatter(X[ward_predict==3,0],X[ward_predict==3,1],s=100, c='yellow')
plt.title('Ward')
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()

plt.scatter(X[complete_predict==0,0],X[complete_predict==0,1],s=100, c='red')
plt.scatter(X[complete_predict==1,0],X[complete_predict==1,1],s=100, c='blue')
plt.scatter(X[complete_predict==2,0],X[complete_predict==2,1],s=100, c='green')
plt.scatter(X[complete_predict==3,0],X[complete_predict==3,1],s=100, c='yellow')
plt.title('Complete')
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()

plt.scatter(X[average_predict==0,0],X[average_predict==0,1],s=100, c='red')
plt.scatter(X[average_predict==1,0],X[average_predict==1,1],s=100, c='blue')
plt.scatter(X[average_predict==2,0],X[average_predict==2,1],s=100, c='green')
plt.scatter(X[average_predict==3,0],X[average_predict==3,1],s=100, c='yellow')
plt.title('Average')
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()

plt.scatter(X[single_predict==0,0],X[single_predict==0,1],s=100, c='red')
plt.scatter(X[single_predict==1,0],X[single_predict==1,1],s=100, c='blue')
plt.scatter(X[single_predict==2,0],X[single_predict==2,1],s=100, c='green')
plt.scatter(X[single_predict==3,0],X[single_predict==3,1],s=100, c='yellow')
plt.title('Single')
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()


dendrogram1 = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Ward')
plt.xlabel("Data")
plt.ylabel("Euclidean")
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()

dendrogram2 = sch.dendrogram(sch.linkage(X, method='complete'))
plt.title('Complete')
plt.xlabel("Data")
plt.ylabel("Euclidean")
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()

dendrogram3 = sch.dendrogram(sch.linkage(X, method='average'))
plt.title('Average')
plt.xlabel("Data")
plt.ylabel("Euclidean")
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()

dendrogram4 = sch.dendrogram(sch.linkage(X, method='single'))
plt.title('Single')
plt.xlabel("Data")
plt.ylabel("Euclidean")
plt.legend(['Cluster 1','Cluster 2','Cluster 3'])
plt.show()

















