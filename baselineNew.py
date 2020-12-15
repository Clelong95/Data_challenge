import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm
import os
import matplotlib.pyplot as plt

# %%

path=".\\data"


def get_slide_number(a):
    nb_slide = int(a[3:6])
    return(nb_slide)

def get_tile_number(a):
    if a[23]=="_":
        nb_tile=int(a[22])
    elif a[24]=="_":
        nb_tile=int(a[22:24])
    else:
        nb_tile = int(a[22:25])
    return(nb_tile)

# Recuperation des données
    
train_folder=path+"\\train_input\\resnet_features_annotated"
test_folder=path+"\\test_input\\resnet_features"

X=np.empty(shape=(0,2051))
for element in os.listdir(train_folder):
    X = np.concatenate((X,np.load(train_folder+"\\"+element)),axis=0)
    
features_test=[]
Id_test=[]
for element in os.listdir(test_folder):
    features_test.append(np.load(test_folder+"\\"+element))
    Id_test.append(element[3:6])

Y = pd.read_csv(path+'\\train_input\\train_tile_annotations.csv')
Y['SlideNumber']= Y["TileName"].apply(get_slide_number)
Y['TileNumber']=Y["TileName"].apply(get_tile_number)
Y= Y.sort_values(by=["SlideNumber","TileNumber"])

label=Y['Target'].as_matrix()

X_train, X_test, label_train, label_test =sklearn.model_selection.train_test_split(X,label,test_size=0.3)
#%%

model = sklearn.svm.SVC(kernel='rbf',probability=True)
C_list = [0.01,0.03,0.1,0.3,1,3,10,30]
param_grid = dict(C=C_list)
grid = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2)
grid_result = grid.fit(X,label)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#%%

RegLog = sklearn.linear_model.LogisticRegression(penalty='l2',C=0.045)
RegLog.fit(X_train,label_train)
print("RegLog_score_train : " + str(RegLog.score(X_train,label_train)))
print("RegLog_score_test : "+ str(RegLog.score(X_test,label_test)))

#SVM = sklearn.svm.SVC(C=10, kernel='rbf',probability=True)
#SVM.fit(X_train,label_train)
#print("SVM_score_train : " + str(SVM.score(X_train,label_train)))
#print("SVM_score_test : "+ str(SVM.score(X_test,label_test)))


# %%

pred_labels= RegLog.predict(X_test)

# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(pred_labels == 1, label_test == 1))
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 0, label_test == 0))
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 1, label_test == 0))
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 0, label_test == 1))
print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))


Precision=TP/(TP+FP)
Recall =TP/(TP+FN)
print("Precision:" + str(Precision))
print("Recall:" + str(Recall))
print("F Score:"+str(2*Precision*Recall/(Precision+Recall)))

#%%
C_list=np.arange(0.01,1,0.1)
P_list=[]
R_list=[]
F_list=[]

for C in C_list:
    RegLog = sklearn.linear_model.LogisticRegression(penalty='l2',C=C)
    RegLog.fit(X_train,label_train)
    pred_labels= RegLog.predict(X_test)
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, label_test == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, label_test == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, label_test == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, label_test == 1))
    Precision=TP/(TP+FP)
    Recall =TP/(TP+FN)
    P_list.append(Precision)
    R_list.append(Recall)
    F_list.append(2*Precision*Recall/(Precision+Recall))

#plt.plot(C_list,P_list,'r')
#plt.plot(C_list,R_list,'g')
plt.plot(C_list,F_list,'b')

#%%

pred_y=RegLog.predict(X_test) 
probs_y=RegLog.predict_proba(X_test)
 

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(label_test, probs_y[:,1]) 
  #retrieve probability of being 1(in second column of probs_y)
f_score= (2*precision*recall)/(precision+recall)

pr_auc = sklearn.metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.plot(thresholds,f_score[: -1], "g--", label="F score")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

M = max(f_score)
m = thresholds[np.argmax(f_score)]
plt.scatter(m,M)
plt.savefig('Precision_Recall.jpg')

#%%
Score=[]
threshold=np.arange(0,1,0.01)
for i in threshold:
    pred = (probs_y[:,1]>i)*1
    Score.append(1 - sklearn.metrics.mean_squared_error(label_test,pred))

plt.plot(threshold,Score)

# %%

# Prediction on the test set

# Train a final model on the full training set
estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.045)
#estimator = sklearn.svm.SVC(C=3.0,probability=True,kernel="rbf")
estimator.fit(X, label)


prediction_test_tiles=[]
for i in range(len(features_test)):
    prediction_test_tiles.append(estimator.predict_proba(features_test[i])[:, 1])

#%%    
prediction_test=[]
for i in prediction_test_tiles:
    prediction_test.append(np.sort(i)[-5])
    
#prediction_test_2=[]
#for i in prediction_test_tiles:
#    prediction_test_2.append(np.mean(np.sort(i)[-5:]))
    
#A=[(prediction_test_1[i]-prediction_test_2[i])**2 for i in range(120)]
# %%
# -------------------------------------------------------------------------
# Write the predictions in a csv file, to export them in the suitable format
# to the data challenge platform
test_output = pd.DataFrame({"ID": Id_test, "Target": prediction_test})
test_output.set_index("ID", inplace=True)
test_output.to_csv(path + "\\preds_test_baseline.csv")
