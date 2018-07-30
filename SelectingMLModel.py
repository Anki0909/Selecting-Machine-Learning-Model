def selectMLmodel(X_training, y_training):
    """
    Selects best performing classification machine learning algorithm for given data.

    Parameters:
    ------------
    X_training : {array_like, sparse matrix},
                shape = [n_samples, n_features]
                Matrix of training samples.

    y_training : array_like, shape = [n_samples]
                Vector of target class labels

    Returns:
    -----------
    Prints classifier name along with its performance on training and validation data and 
    the tuned optimized hyperparameters of the classifier.

    """

    # Imputing missing values if present in the data
    from sklearn.preprocessing import Imputer
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit_transform(X_training)

    # Importing the classifiers with default settings
    from sklearn.linear_model import Perceptron
    ppn = Perceptron()
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    from sklearn.tree import DecisionTreeClassifier
    tree =  DecisionTreeClassifier()
    from sklearn.svm import SVC 
    
    # The default kernel for SVC is 'rbf' or the gaussian kernel, but since the most basic SVM
    # learnt as basic is linear, it is implemented here to test against other algorithm.

    svm_l = SVC(kernel='linear')
    svm_k = SVC()
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()

    classifier_list = [ppn, lr, tree, svm_l, svm_k, knn, rf]
    classifier_label = ['Perceptron', 'Logistic Regression', 'Decision Tree', 'Linear SVM', 'Gaussian SVM', 'K Nearest Neighbour', 'Random Forest']

    # Standardizing the given data
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_training_std = stdsc.fit_transform(X_training)

    # Cross validating
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    scoring = ['accuracy', 'f1_weighted', 'roc_auc']
    print('Cross validation:')
    for classifier, label in zip(classifier_list, classifier_label):
        scores = cross_validate(estimator=classifier,
                                        X=X_training_std,
                                        y=y_training,
                                        cv=10,
                                        scoring=scoring)
        print("[%s]\nAccuracy: %0.3f\tF1 Weighted: %0.3f\tROC AUC: %0.3f"
                % (label, scores['test_accuracy'].mean(), scores['test_f1_weighted'].mean(), scores['test_roc_auc'].mean()))
        