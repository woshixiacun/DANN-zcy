import numpy as np

def knnclassification(Xtest, Xtrain, Ytrain, k=1, norm='2norm'):
    n_test = Xtest.shape[0]
    Ypred = np.zeros(n_test, dtype=int)

    for i in range(n_test):
        diff = Xtrain - Xtest[i]

        if norm == '2norm':
            dist = np.sqrt(np.sum(diff**2, axis=1))
        elif norm == '1norm':
            dist = np.sum(np.abs(diff), axis=1)
        else:
            raise ValueError("Unsupported norm")

        idx = np.argsort(dist)[:k]
        labels = Ytrain[idx]

        # majority vote
        values, counts = np.unique(labels, return_counts=True)
        Ypred[i] = values[np.argmax(counts)]

    return Ypred
