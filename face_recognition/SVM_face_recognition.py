
from sklearn.model_selection import train_test_split
from preprocess import proces_dataset_faces
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
path_faces = "data/preprocess_faces/"

X,Y = proces_dataset_faces(path_faces)

print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


print("X Shape %s " % str(X.shape))
print("Y shape %s " % str(Y.shape))

print("X train Shape %s " % str(X_train.shape))
print("Y shape %s " % str(Y.shape))
n_components_PCA = 50

pca = PCA(n_components=n_components_PCA)
print("Fitting PCA with %s" % str(n_components_PCA))
pca.fit(X_train)
print("PCA fitted! with X_Train ")

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print("X_train and X_test pca-transformed" )
print("X_test shape %s " % str(X_test.shape))
print("X_train shape %s " % str(X_train.shape))

param_grid={ 'kernel': [ 'poly', 'rbf', 'linear', 'sigmoid' ],
             'C' : [ 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e+1,1.0e2,1.0e3,1.0e4 ],
             'degree' : [ 1, 2, 3 ],
             'gamma': [ 0.001,0.01,0.0001 ],
             'coef0' : [ 0.0 ] }


print("Training SVM")

svr = GridSearchCV( SVC( max_iter=10000 ), param_grid )
#print( 'Best regressor for %d ' % i, svr.best_estimator_ )
svr.fit( X_train, Y_train )
y_predict = svr.predict( X_test )
print("Best estimator for : %s "% svr.best_estimator_)
print("Predicting with SVM")

print("-"*50)
print("SVM ")
print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_predict).sum(), len(Y_test) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_predict).sum() ) / len(Y_test) ) )
print("-"*50)

