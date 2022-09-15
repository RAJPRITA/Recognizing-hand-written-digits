

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


#part: library dependencies --sklearn, torch, tensorflow, numpy, transformer
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

#1 .Set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

#exit() 

#MODEL HYPERPARAMS



#2 .For every combination of hyper parameter values


train_frac = 0.1
test_frac = 0.1
dev_frac = 0.1



#part: Load dataset -- data from csv,tsv,jsonl,pickle

digits = datasets.load_digits()

#PART:Sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)



# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=1-train_frac, shuffle=True
)

X_test,X_dev, y_test,y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/(1-train_frac), shuffle=True
)

GAMMA = 0.001
C = 0.5
best_acc = -1.0
best_model = None
best_h_params = None
for cur_h_param in h_param_comb:
    # Create a classifier: a support vector classifier
    #PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #PART: setting up hyperparameter
    hyper_params = cur_h_param ##{'gamma':GAMMA ,'C':C}
    clf.set_params(**hyper_params)

    # Split data into 50% train and 50% test subsets
    #2.a train the model
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    print(cur_h_param)
    #PART: get test dev pridiction
    predicted_dev = clf.predict(X_dev)
    #2.b compute the accuracy on the validation set
    cur_acc =metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    #3. identify the combination of hyper parametsrs for which the validation set accuracy is the highest

    if cur_acc > best_acc :
        best_acc = cur_acc
        best_model = clf
        best_h_params = cur_h_param
        print("Found new best acc with :"+str(cur_h_param))
        print("New best val accuracy :"+str(cur_acc))
      
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.
print(cur_h_param)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )
#4. report the test set accuracy with that best model
print("Best hyperparameters were:")
print(cur_h_param)
###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

#plt.show()


