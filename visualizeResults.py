"""
This program is used to visualize the results of the classifier and helped to try changes and different ideas
during the work on the project.

we experiment with our font classification system, offered by the 'evaluate'
and 'predict' function defined in 'classify.py'. We use the 'test' partition in the
'ReadySynthText.h5' dataset produced by 'dataPrepper.py'. We also use the 'train' partition
for comparison.
The file also includes a few utility functions for calculating quality measures of the
classifier and for plotting results.

The main program tests the system and its models on the above-mentioned dataset, and produces
the following:
    > Test ROC curves for individual classes (plotted using matplotlib).
    > Test confusion matrix (displayed using IPython).
    > Individual test sensitivities for each font (displayed using IPython).
    > Overall test and train accuracies.

"""

from utils import font_index_map, num_of_fonts, PROJECT_PATH
from classify import evaluateWightedVectors
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import h5py
import pandas as pd
from IPython.display import display


def plotROCs(probs, actual, title, num_classes, class_names):
    """
    plotROCs : plot ROC curves for a classifier (using matplotlib.pyplot).
        Input
            > 'probs' - an N-by-'num_classes' numpy matrix whose [i,j] entry is the predicted
                probability for the i'th example being in the j'th class.
            > 'actuals' - a list of N integers, whose i'th element is the index of the class to
                which the i'th example belongs to.
            > 'title' - an optional title for the plot. The title will also include the average AUC.
            > 'num_classes' - the number of classes.
            > 'class_names' - list of class names (whose i'th entry is the name of the i'th class)
    """

    actual = label_binarize(actual, classes=range(num_classes)) # binarize actual labels
    
    FPRs = [] # list to hold true-positive-rates for each class
    TPRs = [] # list to hold false-positive-rates for each class
    AUCs = [] # list to hold AUC scores for each class
    
    for class_i in range(num_classes):
        # compute the necessery information for each class:
        curr_FPRs, curr_TPRs, _ = roc_curve(actual[:, class_i], probs[:, class_i])
        curr_AUC = auc(curr_FPRs, curr_TPRs)
        FPRs.append(curr_FPRs)
        TPRs.append(curr_TPRs)
        AUCs.append(curr_AUC)
    average_AUC = np.average(AUCs) # calculate average AUC

    # plot each ROC curve:
    plt.figure(figsize=(6,6))
    for class_i in range(num_classes):
        plt.plot(FPRs[class_i], TPRs[class_i], label=("ROC of class: " + str(class_names[class_i])
                                                      + "; AUC=" + str(round(AUCs[class_i],3))))
    plt.plot([0, 1], [0, 1], 'k--') # plot ROC of random classifier (for comparison)
    
    # at last - specify a few cosmetics for the plot
    plt.xlim(-0.02, 1.0)
    plt.ylim(0.0, 1.02)
    plt.gca().set_aspect(1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(((title + '\n') if title!=None else "") +
                   "Average AUC: " + str(round(average_AUC,4)))
    plt.legend(prop={'size':10})
    
    
def calcConfusionMat(predicted, actual, num_classes):
    """
    calcConfusionMat : calculates confusion matrix.
        Input
            > 'predicted' - a list of N integers, whose i'th element is the index of the predicted
                class for the i'th example.
            > 'actual' - a list of N integers, whose i'th element is the index of the actual class
                the i'th example belongs to.
            > 'num_classes' - number of classes to distinguish between.
        Returns
            A 'num_classes' by 'num_classes' confusion matrix (as numpy array). Rows represent
            predicted classes and columns represent actual classes.
    """

    # pair predicted classes with actual classses
    confusion_pairs = list(zip(predicted, actual))
    # build the confusion matrix based of the number of occurrences of each pair:
    confusion_matrix = [[confusion_pairs.count((pred_class, act_class))
                         for act_class in range(num_classes)]
                         for pred_class in range(num_classes)]
    return np.array(confusion_matrix)


def calcSensitivities(M):
    """
    calcSensitivities : calculates sensitivity of a classifier to different classes.
        Input
            > 'M' - confusion matrix produced for the classifier.
        Returns
            List of sensitivities (whose i'th element is the sensitivity for the i'th class).
            Sensitivity is calculated as [true_positives / (true_positives+false_negatives)]
    """
    results = []
    # calculate sensitivity (directly) for each class:
    for i in range(M.shape[0]):
        sensitivity = M[i,i] / M[:,i].sum()
        results.append(sensitivity)
    return results


def visualizeResults_main():
    """
    VisualizeResults_main : tests the font classification system on the 'test' and 'train' partition
        in the dataset 'ReadySynthText.h5' produced by 'dataPrepper.py'.
    Requirements: the 'models' directory (produced by 'train.py'), and the 'ReadySynthText.h5'
         dataset (produced by 'dataPrepper.py'), being in the PROJECT_PATH.
    Produces the following quality measures:
        > Test ROC curves for individual classes (plotted using matplotlib).
        > Test confusion matrix (displayed using IPython).
        > Individual test sensitivities for each font (displayed using IPython).
        > Overall test and train accuracies.
    WARNING: the 'ReadySynthText.h5' dataset (produced by 'dataPrepper.py') is very heavy size.
        To avoid producing it yourself - see the results I generated from this program, presented
        and described in detail in the attached report
    """

    # load the dataset
    with h5py.File(PROJECT_PATH + 'ReadySynthText.h5', 'r') as db:
        
        # import necessary information
        test_imgs = db['test']['data']['images']
        test_actuals = db['test']['data']['font_index'][()]
        test_char_indexing = db['test']['indexing']['by_char']
        test_word_indexing = db['test']['indexing']['by_word']
        train_imgs = db['train']['data']['images']
        train_actuals = db['train']['data']['font_index'][()]
        train_char_indexing = db['train']['indexing']['by_char']
        train_word_indexing = db['train']['indexing']['by_word']
        
        # evaluate probabilities and predictions for test and train examples
        print('Making predictions on test examples...')
        test_probs = evaluateWightedVectors(test_imgs, test_char_indexing, test_word_indexing)
        test_preds = np.argmax(test_probs, axis=1)
        print('Done making predictions on test examples.\n')
        print('Making predictions on train examples...')
        train_probs = evaluateWightedVectors(train_imgs, train_char_indexing, train_word_indexing)
        train_preds = np.argmax(train_probs, axis=1)
        print('Done making predictions on train examples.\n')
        
        # decode class names from the font_index map
        class_names = [font.decode("utf-8") for font in font_index_map]
        
        # plot test ROC curves
        plotROCs(test_probs, test_actuals, title='ROC Curves (Test Data)',
                 num_classes=num_of_fonts, class_names=class_names)
        
        # calculate and display confusion matrix for test examples
        test_conf_mat = calcConfusionMat(test_preds, test_actuals, num_classes=num_of_fonts)
        conf_mat_df = pd.DataFrame(test_conf_mat, class_names, class_names)
        print("Confusion Matrix (Test Data)")
        conf_mat_df.style.set_caption("Confusion Matrix (Test Data)")
        display(conf_mat_df)
        print('')
        
        # calculate and display sensitivities for test examples (in descending order)
        sensis = calcSensitivities(test_conf_mat)
        sensis_and_cnames = sorted(zip(sensis, class_names), reverse=True,
                                   key=lambda x:x[0])
        sensis_df = pd.DataFrame([sensi for (sensi,cname) in sensis_and_cnames],
                                 [cname for (sensi,cname) in sensis_and_cnames])
        print("Sensitivities (Test Data)")
        sensis_df.style.set_caption("Sensitivities (Test Data)")
        display(sensis_df)
        print('')
        
        # calculate and display test and train accuracies
        train_acc = np.count_nonzero(train_preds == train_actuals) / len(train_actuals)
        test_acc = np.count_nonzero(test_preds == test_actuals) / len(test_actuals)
        accuracies_df = pd.DataFrame([train_acc, test_acc],
                                     ['train accuracy', 'test accuracy'])
        print("Accuracies")
        accuracies_df.style.set_caption("Accuracies")
        display(accuracies_df)



