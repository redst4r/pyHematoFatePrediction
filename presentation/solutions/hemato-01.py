plt.hist(X_class0.mean((1,2)), bins = np.linspace(0.3,0.9,100), histtype='step', normed=True);
plt.hist(X_class1.mean((1,2)), bins = np.linspace(0.3,0.9,100), histtype='step', normed=True);
plt.xlabel('Intensity')
plt.ylabel('Relative frequency')
plt.legend(['Class1', 'Class2']);