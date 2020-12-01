import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from IPython.display import Image
plt.style.use('seaborn')

# In[9]:


# digits dataset for input samples
def extract_dataset(f):
    data = []
    for line in f:
        if (line.split()[0] == "1.0000"):
            data.append(line.split())
        elif(line.split()[0] == "5.0000"):
            data.append(line.split())

    # changing the values of 5.0000 to -1 and 1.0000 the value of 1
    df = pd.DataFrame(data)
    df[0] = df[0].replace(["1.0000","5.0000"],[1,-1])
    
    for col in df.columns:
        df[col] = df[col].astype(float)

    features, rows, cols = extract_features(df)
    
    return features, rows, cols, df[0].astype(int)

def extract_features(df):
    rows, cols = df.shape
    features = np.zeros(shape = (rows,3))
    
    df1 = df.copy()
    df1 = df1.drop(0,axis=1)
    
    # extracting the intensity from the dataset
    for i in range(rows):
        mean_int = df1.loc[i,0:255].mean()
        features[i][0] = 1
        features[i][1] = mean_int
    

    flipped_df = df.loc[:, :1:-1]
    flipped_df.columns = df1.columns
    diff = pd.DataFrame()
    diff = abs(df1.subtract(flipped_df))
    
    # extracting the symmetry from the dataset
    for i in range(rows):
        mean_sym = diff.loc[i,0:255].mean()
        features[i][0] = 1
        features[i][2] =  mean_sym*-1

    return features, rows, cols


# In[10]:


def bar_plot(cv_error, in_sample_error):
    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(cv_error))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, cv_error, color='steelblue', width=barWidth, edgecolor='white', label='cv_error')
    plt.bar(r2, in_sample_error, color='darksalmon', width=barWidth, edgecolor='white', label='in_sample_error')

    # Add xticks on the middle of the group bars
    plt.ylabel('Error', fontweight='bold')
    plt.xlabel('Regularization parameter', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(cv_error))], [0.01, 0.1, 1, 10, 100])

    # Create legend & Show graphic
    plt.legend()
    plt.show()


# In[11]:


f_train = open("/Users/bharathi/Documents/Digits/ZipDigits.train", "r")
X, rows, cols, y = extract_dataset(f_train)

f_test = open("/Users/bharathi/Documents/Digits/ZipDigits.test", "r")
X_testdata, test_rows, test_cols, y_testdata = extract_dataset(f_test)


# In[230]:



# C = [0.01, 0.1, 1, 10, 100]
# kf = KFold(n_splits=5, shuffle = True, random_state = 10)
# kf.split(X)
# cv_error = {}


# for reg_par in C:
#     scores = []
#     in_sample_error = []
#     print()
#     print("Regularization parameter, C = ",reg_par)
#     clf = SVC(C = reg_par, kernel='linear')
    
# #     clf.fit(X, y)
# #     y_pred = clf.predict(X)

# #     accuracy = sklearn.metrics.accuracy_score(y, y_pred, normalize=True)
# #     in_sample_error[reg_par] = 1 - accuracy
# #     print("In-sample error : ", 1 - accuracy)
    
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf.fit(X_train, y_train)
#         y_pred_train = clf.predict(X_train)
#         y_pred = clf.predict(X_test)
#         in_sample_error.append(1 - clf.score(X_train, y_train))
#         scores.append(1 - clf.score(X_test, y_test))
        
#     print("Errors during each fold: ", scores)
#     cv_error[reg_par] = np.mean(scores)
#     print("CV error : ", np.mean(scores))
# #     bar_plot(in_sample_error, scores)
#     line_plot(in_sample_error, scores)

# print()
# opt_error = min(cv_error.values())
# opt_key = [key for key in overall_scores if cv_error[key] == opt_error]
# print("Minimum error for a linear kernel = ", opt_error, "for C = ", opt_key[0])


# ###SVM classifier using the linear kernel

# In[13]:



C = [0.01, 0.1, 1, 10, 100]
kf = KFold(n_splits=5, shuffle = True, random_state = 10)
kf.split(X)
cv_error = {}
in_sample_error = {}
final_hyp = {}

for reg_par in C:
    scores = []
    print()
    print("Regularization parameter, C = ",reg_par)
    clf = SVC(C = reg_par, kernel='linear')
    
    clf.fit(X, y)
    y_pred = clf.predict(X)

    accuracy = sklearn.metrics.accuracy_score(y, y_pred, normalize=True)
    in_sample_error[reg_par] = 1 - accuracy
    print("In-sample error : ", 1 - accuracy)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(1 - clf.score(X_test, y_test))
        
    print("Errors during each fold: ", scores)
    cv_error[reg_par] = np.mean(scores)
    print("CV error : ", np.mean(scores))

print()
opt_error = min(cv_error.values())
opt_key = [key for key in cv_error if cv_error[key] == opt_error]
print("Minimum error for a linear kernel = ", opt_error, "for C = ", opt_key[0])
final_hyp['linear'] = {'reg_par' : opt_key[0], 'error' : opt_error}

bar_plot(cv_error.values(), in_sample_error.values())
# line_plot(in_sample_error, cv_error)


# ###SVM classifier using the RBF kernel

# In[14]:



C = [0.01, 0.1, 1, 10, 100]
kf = KFold(n_splits=5, shuffle = True, random_state = 10)
kf.split(X)
cv_error = {}
in_sample_error = {}

for reg_par in C:
    scores = []
    print()
    print("Regularization parameter, C = ",reg_par)
    clf = SVC(C = reg_par, kernel='rbf')
    
    clf.fit(X, y)
    y_pred = clf.predict(X)

    accuracy = sklearn.metrics.accuracy_score(y, y_pred, normalize=True)
    in_sample_error[reg_par] = 1 - accuracy
    print("In-sample error : ", 1 - accuracy)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(1 - clf.score(X_test, y_test))
        
    print("Errors during each fold: ", scores)
    cv_error[reg_par] = np.mean(scores)
    print("CV error : ", np.mean(scores))

print()
opt_error = min(cv_error.values())
opt_key = [key for key in cv_error if cv_error[key] == opt_error]
print("Minimum error for an RBF kernel = ", opt_error, "for C = ", opt_key[0])
final_hyp['rbf'] = {'reg_par' : opt_key[0], 'error' : opt_error}

bar_plot(cv_error.values(), in_sample_error.values())
# line_plot(in_sample_error, cv_error)


# ###SVM classifier using the polynomial kernel with degree=3

# In[16]:


C = [0.01, 0.1, 1, 10, 100]
kf = KFold(n_splits=5, shuffle = True, random_state = 10)
kf.split(X)
cv_error = {}
in_sample_error = {}

for reg_par in C:
    scores = []
    print()
    print("Regularization parameter, C = ",reg_par)
    clf = SVC(C = reg_par, kernel='poly', degree = 3)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    accuracy = sklearn.metrics.accuracy_score(y, y_pred, normalize=True)
    in_sample_error[reg_par] = 1 - accuracy
    print("In-sample error : ", 1 - accuracy)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(1 - clf.score(X_test, y_test))
        
    print("Errors during each fold: ", scores)
    cv_error[reg_par] = np.mean(scores)
    print("CV error : ", np.mean(scores))

print()
opt_error = min(cv_error.values())
opt_key = [key for key in cv_error if cv_error[key] == opt_error]
print("Minimum error for a polynomial kernel = ", opt_error, "for C = ", opt_key[0])
final_hyp['poly'] = {'reg_par' : opt_key[0], 'error' : opt_error}

bar_plot(cv_error.values(), in_sample_error.values())
# line_plot(in_sample_error, cv_error)


# In[17]:


print(final_hyp)


# In[18]:


final_hypothesis = min(final_hyp, key=lambda k: final_hyp[k]['error'])
print("Kernel = ", final_hypothesis, "Minimum error = ", final_hyp[final_hypothesis])


clf_final = SVC(C = 100, kernel='poly', degree = 3)
clf_final.fit(X, y)
y_pred = clf_final.predict(X)
overall_error = 1 - clf_final.score(X, y)

print("In-sample error using the final hypothesis = ", overall_error)

y_pred_test = clf_final.predict(X_testdata)
overall_error = 1 - clf_final.score(X_testdata, y_testdata)

print("Test error using the final hypothesis = ", overall_error)

# Image(filename='Prob6.png')

