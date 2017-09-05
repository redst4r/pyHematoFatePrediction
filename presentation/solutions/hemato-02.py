"prediction"
yhat = CNN.predict([X_test, mov_test], batch_size=128, verbose=1)

"score histogram"
plt.hist(yhat[:,1],100);
plt.xlabel('Prediction Score');
plt.ylabel('Frequency');

"Confusion matrix"
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
print(confusion_matrix(np.argmax(yhat,1), np.argmax(y_test, 1)))

# pretty print
from talk_utils import plot_confusion_matrix
plt.figure()
plot_confusion_matrix(yhat,  y_test, classes=[0,1]);


"ROC curve/AUC"
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, 1), yhat[:,1], drop_intermediate=False)
the_auc = auc(fpr, tpr)
print(the_auc)

plt.figure()
plt.scatter(fpr, tpr, c=thresholds, cmap=plt.cm.bwr);
plt.plot(fpr,tpr,'k');

# pretty print
from talk_hemato_utils import get_auc
get_auc(yhat[:,1], y_test[:,1], do_plot=True)