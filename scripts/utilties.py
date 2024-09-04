from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import json


def plot_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred, labels, title):
    """
    Plots a confusion matrix based on true and predicted labels.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : list of str
        List of class labels.
    """
    # Generate confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Plot confusion matrix for train data
    # Visualize confusion matrix using heatmap
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues' , cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Train Data')

    # Plot confusion matrix for test data
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.suptitle('Confusion Matrices for ' + title + ' Model', fontsize=16)
    plt.tight_layout()
    
    plt.show()


# The ROC curve plots the True Positive Rate (TPR) on the y-axis against the False Positive Rate (FPR) on the x-axis as the discrimination threshold is varied.
#  - It helps visualize the model's ability to distinguish between fraudulent and legitimate transactions.
#  - A model with a higher AUC (Area Under the Curve) is generally better at discrimination.

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_train, y_train_pred_prob, y_test, y_test_pred_prob, title):
    """
    Plots the ROC curve for both train and test data.

    Parameters:
    y_train : array-like of shape (n_samples,)
        True labels for training data.
    y_train_pred_prob : array-like of shape (n_samples,)
        Predicted probabilities for training data.
    y_test : array-like of shape (n_samples,)
        True labels for test data.
    y_test_pred_prob : array-like of shape (n_samples,)
        Predicted probabilities for test data.
    title : str
        Title for the plots.
    """
    # Calculate ROC curve and AUC for train data
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_prob)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Calculate ROC curve and AUC for test data
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_prob)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curve for train data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_train, tpr_train, label='ROC curve (area = %0.2f)' % roc_auc_train)
    plt.plot([0, 1], [0, 1], 'k--', label='No Discrimination')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Train Data')
    plt.legend(loc="lower right")

    # Plot ROC curve for test data
    plt.subplot(1, 2, 2)
    plt.plot(fpr_test, tpr_test, label='ROC curve (area = %0.2f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], 'k--', label='No Discrimination')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Test Data')
    plt.legend(loc="lower right")

    plt.suptitle('ROC Curves for ' + title + ' Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Precision-Recall Curve(Use for Imbalanced DataSet):

# - The precision-recall curve plots Precision (positive predictive value) on the y-axis against Recall (true positive rate) on the x-axis as the classification threshold is varied.
# - This is useful when dealing with imbalanced datasets, where positive cases (fraudulent transactions) might be rare.
# - A model with a curve that stays closer to the top-left corner indicates a better balance between precision and recall.
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(y_train, y_train_pred_prob, y_test, y_test_pred_prob, title):
    """
    Plots the precision-recall curve for both train and test data.

    Parameters:
    y_train : array-like of shape (n_samples,)
        True labels for training data.
    y_train_pred_prob : array-like of shape (n_samples,)
        Predicted probabilities for training data.
    y_test : array-like of shape (n_samples,)
        True labels for test data.
    y_test_pred_prob : array-like of shape (n_samples,)
        Predicted probabilities for test data.
    title : str
        Title for the plots.
    """
    # Calculate precision-recall curve for train data
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_prob)
    
    # Calculate precision-recall curve for test data
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob)

    # Plot precision-recall curve for train data
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(recall_train, precision_train, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Train Data')
    plt.legend(loc="lower left")

    # Plot precision-recall curve for test data
    plt.subplot(1, 2, 2)
    plt.plot(recall_test, precision_test, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Test Data')
    plt.legend(loc="lower left")

    # Add main title
    plt.suptitle('Precision-Recall Curves for ' + title + ' Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()





# Distribution Plots:

# - Create histograms or kernel density estimation (KDE) plots to visualize the distribution of features or predicted probabilities for both fraudulent and legitimate transactions.
# - This can help identify potential patterns or outliers that might be related to fraudulent activity.

def plot_distribution(data, feature_name, class_label="class_label", kind="kde"):
  """
  Plots the distribution of a feature for different classes.

  Parameters:
    data : pandas DataFrame
      DataFrame containing the data.
    feature_name : str
      Name of the feature to plot.
    class_label : str, optional
      Name of the class label column (default: "class_label").
    kind : str, optional
      Plot kind (e.g., "hist" for histogram, "kde" for kernel density estimation).
  """
  sns.displot(data=data, x=feature_name, hue=class_label, kind=kind)
  plt.title(f'Distribution of {feature_name} by {class_label}')
  plt.show()


def plot_predicted_probability_distribution(y_train_pred_prob, y_test_pred_prob, bins=10, title=''):
    """
    Plots the distribution of predicted probabilities for both train and test data.

    Parameters:
    y_train_pred_prob : array-like
        Array of predicted probabilities for training data.
    y_test_pred_prob : array-like
        Array of predicted probabilities for test data.
    bins : int, optional
        Number of bins for the histogram (default: 10).
    title : str
        Title for the plots.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot distribution for train data
    plt.subplot(1, 2, 1)
    sns.histplot(y_train_pred_prob, bins=bins, kde=True, color='blue')
    plt.title('Distribution of Predicted Probabilities - Train Data')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    
    # Plot distribution for test data
    plt.subplot(1, 2, 2)
    sns.histplot(y_test_pred_prob, bins=bins, kde=True, color='green')
    plt.title('Distribution of Predicted Probabilities - Test Data')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    
    # Add main title
    plt.suptitle('Distributions of Predicted Probabilities for ' + title + ' Model', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()



def show_result(y_train, y_train_pred, y_test, y_test_pred, labels,  title, sampling_type):
    # Calculate train metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)

    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Train Precision: {train_precision:.2f}")
    print(f"Train Recall: {train_recall:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")

   
   # f1score = f1_score(precision, recall)
    #print(f"F1-Score::  {f1score:.2f}")
    
  
    plot_confusion_matrix(y_train, y_train_pred, y_test, y_test_pred, labels, title)
    plot_roc_curve(y_train, y_train_pred, y_test, y_test_pred, title)
    plot_predicted_probability_distribution(y_train_pred, y_test_pred,  title=title)
    
    return {
        "train": {"accuracy": train_accuracy, "precision": train_precision, "recall": train_recall},
        "test": {"accuracy": test_accuracy, "precision": test_precision, "recall": test_recall},
        "sampling": sampling_type
    }


def print_comparision_result(comparision_result):
   return json.dumps(comparision_result, indent = 1)


