from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from preprocess import proces_dataset_faces

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

classifier = KNeighborsClassifier( n_neighbors=3, weights='uniform' )
print("Fitting KNN Classifier")
classifier.fit( X_train, Y_train )
print("Predicting with KNN Casiffier")
y_pred = classifier.predict( X_test )

print("-"*50)
print("KNN ")
print( "%d muestras mal clasificadas de %d" % ( (Y_test != y_pred).sum(), len(Y_test) ) )
print( "Accuracy = %.1f%%" % ( ( 100.0 * (Y_test == y_pred).sum() ) / len(Y_test) ) )
print("-"*50)


