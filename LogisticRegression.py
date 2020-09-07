import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


data = load_breast_cancer()
print(data.DESCR)

# Explanatory variables
X = data['data']
print("Feature Names"+ str(data['feature_names']))
# Response variable.
# Relabel such that 0 = 'benign' and 1 = malignant.
Y = 1 - data['target']
label = list(data['target_names'])
label.reverse()
print("Target Names"+ str(label))

# Visualize the frequency table
ser = pd.Series(Y)
table = ser.value_counts()
table = table.sort_index()
sns.barplot(label,table.values)
plt.show()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1234)

LL = LogisticRegression(solver='liblinear',max_iter=200)
LL.fit(X_train,Y_train)
Y_pred_test = LL.predict(X_test)

Confusion_matrix = metrics.confusion_matrix(Y_test,Y_pred_test)
print(Confusion_matrix)

acc = (Confusion_matrix[0,0] + Confusion_matrix[1,1])/np.sum(Confusion_matrix)
sens = Confusion_matrix[1,1]/(Confusion_matrix[1,0]+Confusion_matrix[1,1])
spec = Confusion_matrix[0,0]/(Confusion_matrix[0,0]+Confusion_matrix[0,1])
prec = Confusion_matrix[1,1]/(Confusion_matrix[0,1]+Confusion_matrix[1,1])
print('Accuracy    = {}'.format(np.round(acc,3)))
print('Sensitvity  = {}'.format(np.round(sens,3)))
print('Specificity = {}'.format(np.round(spec,3)))
print('Precision   = {}'.format(np.round(prec,3)))

#the probability of Y  = 1 prediction

Y_pred_test_prob=LL.predict_proba(X_test)[:,1]

cutoff = 0.7                                                      # cutoff can be a value between 0 and 1.
Y_pred_test_val = (Y_pred_test_prob > cutoff).astype(int)
Confusion_matrix = metrics.confusion_matrix(Y_test,Y_pred_test_val)
print(Confusion_matrix)

acc = (Confusion_matrix[0,0] + Confusion_matrix[1,1])/np.sum(Confusion_matrix)
sens = Confusion_matrix[1,1]/(Confusion_matrix[1,0]+Confusion_matrix[1,1])
spec = Confusion_matrix[0,0]/(Confusion_matrix[0,0]+Confusion_matrix[0,1])
prec = Confusion_matrix[1,1]/(Confusion_matrix[0,1]+Confusion_matrix[1,1])
print('Accuracy    = {}'.format(np.round(acc,3)))
print('Sensitvity  = {}'.format(np.round(sens,3)))
print('Specificity = {}'.format(np.round(spec,3)))
print('Precision   = {}'.format(np.round(prec,3)))

#ROC curve

cutoff_grid = np.linspace(0.0,1.0,100)
TPR = []
FPR = []
for cutoff in cutoff_grid:
    Y_pred_test_val = (Y_pred_test_prob > cutoff).astype(int)
    Confusion_matrix = metrics.confusion_matrix(Y_test,Y_pred_test_val)
    sens = Confusion_matrix[1,1]/(Confusion_matrix[1,0]+Confusion_matrix[1,1])
    spec = Confusion_matrix[0,0]/(Confusion_matrix[0,0]+Confusion_matrix[0,1])
    TPR.append(sens)
    FPR.append(1-spec)

#Visualize it

plt.plot(FPR,TPR,c='red',linewidth=1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#AUC calculation
auc = metrics.roc_auc_score(Y_test,Y_pred_test_prob)
print('AUC  = {}'.format(np.round(auc,3)))