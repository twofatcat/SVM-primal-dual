input_file = '/Users/cao.yumin/Downloads/assignment1-data/data/train.csv'
train_file = '/Users/cao.yumin/Downloads/assignment1-data/data/test.csv'

import cvxopt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# load the data
df = pd.read_csv(input_file, names=['label'] + [i for i in range(200)])
# print(df)
# type of label should be int instead of float
df.loc[:, 'label'].replace(0.0, -1, inplace=True)
df.loc[:, 'label'].replace(1.0, 1, inplace=True)
# df['label'] = df['label'].apply(np.int64)
# print(df)

# # scale the data
X = df.iloc[:, 1:]
Y = df.iloc[:, 0:1]
scaler = StandardScaler()
df.iloc[:, 1:] = scaler.fit_transform(X)
# print(df)
X_data = df.iloc[:, 1:].to_numpy()
Y_data = Y.to_numpy().reshape(-1, 1)
# print(X_data)
# print(Y_data)
#

# crossvalidation set: 1/2 * n_sample, kfold: split into 4
num_rows, num_cols = X.shape
Cv_df = df.iloc[:int(num_rows / 2), :]
kf = KFold(n_splits=3, shuffle=True, random_state=42)
# store the index when using the kfold
train_list = []
cross_list = []
for train, cross in kf.split(Cv_df):
    # print('train>>',train,'len',len(train),'test>>',cross,len(cross))
    train_list.append(train)
    cross_list.append(cross)
Cv_X = X_data[:int(num_rows / 2), :]
Cv_Y = Y_data[:int(num_rows / 2), :]


# print(train_list)
# print(cross_list)
# print(Y_data.shape)
# print('>>>>>>>>>>>>')
# print(Y_data[train_list[1]].shape)
class Primal:
    '''
    the form of label_train: np.array([[1],[2],[3]...])
    the form of data_train: np.array([[1,2,3],[4,5,6]...])
    '''

    def __init__(self, x_data):
        self.X_data = x_data
        self.n_sample, self.n_feature = self.X_data.shape
        self.sigma_primal = []

    def svm_train_primal(self, data_train, label_train, regularisation_para_C=1):  # cvxpot X = [b,w,sigma].T
        tmp1 = -1 * label_train
        tmp2 = -1 * np.array([label_train[i] * data_train[i].T for i in range(self.n_sample)])
        tmp3 = -1 * np.identity(self.n_sample)
        cond1 = np.hstack((tmp1, tmp2, tmp3))
        tp1 = np.zeros(self.n_sample).reshape(-1, 1)
        tp2 = np.zeros(self.n_sample * self.n_feature).reshape(self.n_sample, self.n_feature)
        tp3 = np.identity(self.n_sample)
        cond2 = -1 * np.hstack((tp1, tp2, tp3))
        G = cvxopt.matrix(np.vstack((cond1, cond2)))

        h = cvxopt.matrix(np.hstack((-1 * np.ones(self.n_sample), np.zeros(self.n_sample))))
        # A = cvxopt.matrix(np.zeros(1+self.n_feature+self.n_sample),(1,1+self.n_feature+self.n_sample),'d')
        # b = cvxopt.matrix(0.0)

        tmp4 = 0 * np.identity(1 + self.n_sample + self.n_feature)
        tmp4[1:self.n_feature + 1, 1:self.n_feature + 1] = np.identity(self.n_feature)
        P = cvxopt.matrix(tmp4)

        q = cvxopt.matrix(np.hstack((np.zeros(1), np.zeros(self.n_feature),
                                     regularisation_para_C * np.ones(self.n_sample) / self.n_sample)))
        solution = cvxopt.solvers.qp(P, q, G, h)
        # solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        X = np.ravel(solution['x'])

        w_primal = X[1:self.n_feature + 1]
        b_primal = X[0]
        self.sigma_primal = X[self.n_feature + 1:]
        return w_primal, b_primal, self.sigma_primal

    def svm_predict_primal(self, data_test, label_test, svm_model):
        w = svm_model[0]
        b = svm_model[1]
        return np.sign(np.dot(data_test, w) + b)

    def accuracy(self, label_test, y_predict):  # y_predict is the object of svm_predict_primal
        y_test = label_test
        y_pred = y_predict
        count = 0
        for i in range(len(y_test)):
            if abs(y_pred[i] - y_test[i]) < 1e-5:
                # if (y_pred[i]*y_test[i]) >= (1 - self.sigma_primal[i]): # with error taken into consideration.
                count += 1
        final_accuracy = count / len(y_pred)
        return final_accuracy


class Dual:
    def __init__(self, x_data):
        self.x_data = x_data
        self.n_sample, self.n_feature = self.x_data.shape

    def svm_train_dual(self, data_train, label_train, regularisation_para_C=1):  # default value
        # K = self.kernal()
        tmp_p = self.P(label_train)
        # P = cvxopt.matrix(np.outer(label_train,label_train)*K)
        P = cvxopt.matrix(tmp_p)
        q = cvxopt.matrix(np.ones(self.n_sample) * -1)
        # A = cvxopt.matrix(label_train.reshape(1,-1).tolist()[0])
        A = cvxopt.matrix(label_train, (1, self.n_sample), 'd')
        # print('>>>A',A.size)
        b = cvxopt.matrix(0.0)

        tmp1 = -1 * np.identity(self.n_sample)
        tmp2 = np.identity(self.n_sample)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))

        tmp3 = np.zeros(self.n_sample)
        tmp4 = (regularisation_para_C / self.n_sample) * np.ones(self.n_sample)
        h = cvxopt.matrix(np.hstack((tmp3, tmp4)))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])  # lagrange

        Alpha = alpha > 1e-5  # alpha can't be zero
        index_Alpha = np.arange(len(alpha))[Alpha]
        alpha_dual = alpha[Alpha]
        X_dual = data_train[Alpha]
        y_dual = label_train[Alpha]

        # calculate weight using: w = sum(alpha_i * y_i * X_i)
        w_dual = np.zeros(self.n_feature)
        for k in range(len(alpha_dual)):
            w_dual += alpha_dual[k] * y_dual[k] * X_dual[k]

        w_dual_for_b = np.array(w_dual)
        # calculate intercept using: b = 1/|S| * sum(1/ys - sum(alpha_i * y_i * X_i.T * X_s))
        b_dual = 0.0
        for i in range(len(alpha_dual)):
            b_dual += 1 / y_dual[i]
            b_dual -= np.dot(w_dual_for_b, X_dual[i].T)
        b_dual /= len(alpha_dual)
        return w_dual, b_dual

    def svm_predict_dual(self, data_test, label_test, svm_model_d):
        w = svm_model_d[0]
        b = svm_model_d[1]
        return np.sign(np.dot(data_test, w) + b)

    def accuracy(self, label_test, y_pred):
        y_test = label_test
        count = 0
        for i in range(len(y_test)):
            if abs(y_pred[i] - y_test[i]) < 1e-5:  # they can be seen same under this condition
                count += 1
        final_accuracy = count / len(y_pred)
        return final_accuracy

    def P(self, label_train):  # dual cvxopt P
        K = 0 * np.identity(self.n_sample)
        for i in range(self.n_sample):
            Xi = self.x_data[i]
            yi = label_train[i]
            for j in range(self.n_sample):
                Xj = self.x_data[j]
                yj = label_train[j]
                # power of || xi-xj ||
                norm_length = yi * yj * np.dot(Xi, Xj.T)
                K[i][j] = norm_length
        return K


C_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # using C/n as penalty

primal_accuracy_list = []
# #
# primal
for C in C_list:
    accuracy = 0
    for i in range(3):  # kford
        X_train = Cv_X[train_list[i]]
        y_train = Cv_Y[train_list[i]]
        X_test = Cv_X[cross_list[i]]
        y_test = Cv_Y[cross_list[i]]
        # print(X_train, X_train.shape)
        # print(y_train, y_train.shape)
        svm_p = Primal(X_train)
        svm_model = svm_p.svm_train_primal(X_train, y_train, regularisation_para_C=C)
        dual_ypred = svm_p.svm_predict_primal(X_test, y_test, svm_model)
        accuracy += svm_p.accuracy(y_test, dual_ypred)
    accuracy /= 3
    primal_accuracy_list.append((accuracy, C))

best = sorted(primal_accuracy_list, reverse=True)[0]
best_C = best[1]

if best_C == 1:
    new_c_list = [i for i in range(1, 11)]
elif best_C == 100:
    new_c_list = [i for i in range(90, 101)]
else:
    new_c_list = [i for i in range(best_C - 9, best_C + 10)]
primal_accuracy_list = []
for C in new_c_list:
    accuracy = 0
    for i in range(3):  # kford
        X_train = Cv_X[train_list[i]]
        y_train = Cv_Y[train_list[i]]
        X_test = Cv_X[cross_list[i]]
        y_test = Cv_Y[cross_list[i]]
        # print(X_train, X_train.shape)
        # print(y_train, y_train.shape)
        svm_p = Primal(X_train)
        svm_model = svm_p.svm_train_primal(X_train, y_train, regularisation_para_C=C)
        dual_ypred = svm_p.svm_predict_primal(X_test, y_test, svm_model)
        accuracy += svm_p.accuracy(y_test, dual_ypred)
    accuracy /= 3
    primal_accuracy_list.append((accuracy, C))

best3 = sorted(primal_accuracy_list, reverse=True)[0]
best_C = best3[1]
print('training accuracy is:', best3[0])
#
# # the dual form should has the same C with primal form.
# # calculate C at first, and this can reduce the a burden of dual problem.

test_df = pd.read_csv(train_file, names=['label'] + [i for i in range(200)])
# print(df)
# type of label should be int instead of float
test_df.loc[:, 'label'].replace(0.0, -1, inplace=True)
test_df.loc[:, 'label'].replace(1.0, 1, inplace=True)
# df['label'] = df['label'].apply(np.int64)
# print(df)

# scale the data
test_X = test_df.iloc[:, 1:]
test_Y = test_df.iloc[:, 0:1]
scaler = StandardScaler()
test_df.loc[:, 1:] = scaler.fit_transform(test_X)
# print(df)
test_X_data = test_df.iloc[:, 1:].to_numpy()
test_Y_data = test_Y.to_numpy().reshape(-1, 1)
int_test_Y_data = test_Y_data.astype(int)

svm_p = Primal(X_data)
svm_model = svm_p.svm_train_primal(X_data, Y_data, regularisation_para_C=best_C)
primal_ypred = svm_p.svm_predict_primal(test_X_data, test_Y_data, svm_model)
test_primal_accuracy = svm_p.accuracy(test_Y_data, primal_ypred)
print('The accuracy of primal form is:', test_primal_accuracy)
print('w is:', svm_model[0], 'b is:', svm_model[1])

print("Prediction report of Primal:")
print(classification_report(int_test_Y_data, primal_ypred))
#

svm_d = Dual(X_data)
svm_model_d = svm_d.svm_train_dual(X_data, Y_data, best_C)
dual_ypred = svm_d.svm_predict_dual(test_X_data, test_Y_data, svm_model_d)
test_dual_accuracy = svm_d.accuracy(test_Y_data, dual_ypred)
print('The accuracy of dual form is:', test_dual_accuracy)
print('w is:', svm_model_d[0], 'b is:', svm_model_d[1])

clf = SVC(C=1.0,
          kernel='rbf',
          gamma='scale',
          tol=0.001,
          class_weight=None,
          max_iter=-1)
steps = [('scaler', StandardScaler()), ('svm', clf)]
pipeline = Pipeline(steps)
parameters = {'svm__C': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
gs_clf = GridSearchCV(pipeline, parameters, cv=3)
# print(X_data,X_data.shape)
# print(np.ravel(Y_data).astype(int))
gs_clf.fit(X_data, np.ravel(Y_data).astype(int))
print(gs_clf.best_params_)
y_true, y_pred = int_test_Y_data, gs_clf.predict(test_X_data)
sk_count = 0
for i in range(len(y_true)):
    if abs(y_true[i] - y_pred[i]) < 1e-5:
        sk_count += 1
sk_accuracy = sk_count / len(y_true)
print('The accuracy of sklearn form is:', sk_accuracy)

# print("Prediction report of sklearn:")
# print(classification_report(y_true, y_pred))

