from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

Xf = X[y != 0, :2]
yf = y[y != 0]

#print(X)
#print(y)
#print(Xf)
#print(yf)

plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Iris — classes 1 and 2, sepal features only')
plt.show()





