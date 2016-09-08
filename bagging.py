from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np                                                 
import matplotlib.pyplot as plt                                    
from sklearn.datasets import make_blobs

# create data
centers = [(1, 1), (3, 1)]
X, y = make_blobs(n_samples=10, n_features=2, centers=centers, 
                  cluster_std=0.5, random_state=0)
# training
tr = DecisionTreeClassifier(random_state=0)
bg = BaggingClassifier(DecisionTreeClassifier(random_state=0), 
                       n_estimators=100, random_state=0)
tr.fit(X, y)
bg.fit(X, y)                                                       

# make plot                                                        
fig = plt.figure(frameon=False)                                    
x1_min, x1_max = -1, 5
x2_min, x2_max = -1, 5                                             
h = 0.02
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),               
                    np.arange(x2_min, x2_max, h))
# decision boundary of decision tree
Z = tr.predict((np.array([xx1.ravel(), xx2.ravel()]).T))          
Z = Z.reshape(xx1.shape)
plt.axes().annotate('decision tree', xy=(1.6, 3), xytext=(-0.8, 4),
            arrowprops=dict(facecolor='black', shrink=0.05),       
                            size=16)
plt.contour(xx1, xx2, Z, levels=[0.5], linestyles='dashed')         
# decision boundary of bagged trees
Z = bg.predict((np.array([xx1.ravel(), xx2.ravel()]).T))           
Z = Z.reshape(xx1.shape)
plt.axes().annotate('bagged trees', xy=(2.2, 3), xytext=(3, 4), 
            arrowprops=dict(facecolor='black', shrink=0.05),       
                            size=16)
plt.contour(xx1, xx2, Z, levels=[0.5], linestyles='solid')        

plt.scatter(X[y==0,0], X[y==0, 1], marker='o', s=80, c='white')   
plt.scatter(X[y==1,0], X[y==1, 1], marker='o', s=80, c='black')  
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$x_2$', fontsize=18)                                   
plt.xlim(x1_min, x1_max)
plt.ylim(x1_min, x1_max)                                           
plt.axes().set_aspect('equal')                                     
plt.show()
#plt.savefig('two_blobs.png', bbox_inches='tight')

