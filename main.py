from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.utils import resample
from sklearn.decomposition import PCA
import collections
import pandas as pd
import numpy as np

def imputation(trainX, valX):
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    trainX = imp.fit_transform(trainX)
    valX = imp.transform(valX)
    return trainX, valX

def standidization(trainX, valX):
    scaler = StandardScaler()
    trainX = scaler.fit_transform(trainX)
    valX = scaler.transform(valX)
    return trainX, valX
    
def balance_classes(X, Y):
    Y_index_majority = np.where(Y==0)[0]
    Y_index_minority = np.where(Y==1)[0]
    Y_index_majority_downsampled = resample(Y_index_majority, replace=False, n_samples=Y_index_minority.shape[0]*2, random_state=0)
    Y_index_balanced = np.concatenate((Y_index_majority_downsampled, Y_index_minority), axis=0)
    bal_X = X[Y_index_balanced, :]
    bal_Y = Y[Y_index_balanced]

    return bal_X, bal_Y

def read_file(file_name):
    # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # nemerical_feat = df.select_dtypes(include=numerics)
    df = pd.read_csv(file_name)
    y = df['IsBadBuy'].as_matrix()
    numeric_df = df._get_numeric_data()
    # numeric_headers = list(numeric_df.columns)
    useful_nemerical_feat=['VehicleAge',
                           'WheelTypeID', 
                           'VehOdo', 
                           # 'MMRAcquisitionAuctionAveragePrice',  
                           # 'MMRAcquisitionAuctionCleanPrice',
                           # 'MMRAcquisitionRetailAveragePrice',
                           # 'MMRAcquisitonRetailCleanPrice',
                           'MMRCurrentAuctionAveragePrice',
                           # 'MMRCurrentAuctionCleanPrice',
                           'MMRCurrentRetailAveragePrice',  
                           # 'MMRCurrentRetailCleanPrice'
                           ]
    X = numeric_df[useful_nemerical_feat].as_matrix()
    return X, y

def print_data(trainX, valX, trainY, valY):
    print('Number of training samples: %d'%trainX.shape[0])
    print(collections.Counter(trainY))
    print('Number of validation samples: %d'%valX.shape[0])
    print(collections.Counter(valY))
    print('Feature dimension: %d'%trainX.shape[1])
    print('\n')

def evaluate(Y, Y_pred, dataset):
    print(dataset)
    print('==============')
    print('Accuracy:   %.3f%%'%(accuracy_score(Y, Y_pred)*100))
    P,R,F,_ = precision_recall_fscore_support(Y, Y_pred)
    print('            c0\t\tc1')
    print('Precision:  %.3f\t%.3f'%(P[0], P[1]))
    print('Recall:     %.3f\t%.3f'%(R[0], R[1]))
    print('F-score:    %.3f\t%.3f'%(F[0], F[1]))
    avg_fscore = np.mean(F)
    print("Averaged F-score: %.3f\n"%(avg_fscore))
    return avg_fscore

if __name__ == '__main__':

    # data loading
    train_path = './training.csv'
    test_path = './test.csv'
    X, Y = read_file(train_path)

    # train, validation split
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=41)

    # preprocessing
    trainX, valX = imputation(trainX, valX)
    trainX, trainY = balance_classes(trainX, trainY)
    trainX, valX = standidization(trainX, valX)
    print_data(trainX, valX, trainY, valY)
    

    # initial parameters
    lambdas = np.logspace(-5,1,5)
    best_lambda = None
    best_avg_fscore = None


    for l in lambdas:
        print('Lambda = %f'%l)
        clf = LogisticRegression(C=1./l, verbose=0, class_weight='balanced')
        # , class_weight='balanced'
        # clf = SVC(C=1.0/l, verbose=1, class_weight='balanced', kernel='linear')
        # clf = RandomForestClassifier(class_weight='balanced')
        clf.fit(trainX,trainY)
        trainY_pred = clf.predict(trainX)
        valY_pred = clf.predict(valX)
        avg_fscore = evaluate(trainY, trainY_pred, 'Train')
        avg_fscore = evaluate(valY, valY_pred, 'Validation')

        # update best lambda
        if best_avg_fscore==None or avg_fscore>best_avg_fscore:
            best_lambda = l
            best_avg_fscore = avg_fscore
            print('Find better lambda = %f'%best_lambda)
        print('\n')

    print('Best lambda = %f'%best_lambda)
    print('Best averaged F-score = %.3f'%best_avg_fscore)
