from itertools import product
xgrid = np.array(list(product(np.linspace(-6,6,50),
                              np.linspace(-6,6,50))))
ygrid = MLP.predict(xgrid)

plt.pcolor(xgrid[:,0].reshape(50,50), xgrid[:,1].reshape(50,50), ygrid[:,1].reshape(50,50), cmap=cm.bwr, clim=[0,1])
plt.colorbar()
plt.scatter(X[:,0], X[:,1], c=y, alpha=0.3, cmap=plt.cm.bwr, edgecolors='k'); 
plt.xlabel('x1'), plt.ylabel('x2');