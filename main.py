import numpy as np
from graph import *
from ml import *
from data import *
import sys
from sklearn.metrics import *

print("Tensorflow version "+tf.__version__)


def calculate_metrics(predicted, actual):
    TP = np.count_nonzero(predicted * actual)
    TN = np.count_nonzero((predicted - 1) * (actual - 1))
    FP = np.count_nonzero(predicted * (actual - 1))
    FN = np.count_nonzero((predicted - 1) * actual)

    accuracy = (TP + TN)/ (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    """
       python3 main.py # load model files if available, else retrain and save
       python3 main.py retrain # retrain model and save
       python3 main.py retrain no-save # retrain the model but don't save model
       python3 main.py visualize # load model files and visualize predictions
    """
    retrain_arg = len(sys.argv) > 1 and (sys.argv[1].lower() == 'retrain')
    visualize_arg = len(sys.argv) > 1 and (sys.argv[1].lower() == 'visualize')
    no_save_arg = not (len(sys.argv) > 2 and (sys.argv[2].lower() == 'no-save'))

    train_images, train_labels, test_images, test_labels, class_names = load_data()
    model = ml_model(train_images, train_labels, retrain_arg, no_save_arg)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    train_loss, train_acc = model.evaluate(train_images, train_labels)

    predictions_test = model.predict(test_images)
    predictions_train = model.predict(train_images)

    pred_train = np.argmax(predictions_train, axis=1)
    pred_test = np.argmax(predictions_test, axis=1)
    actual_train = train_labels
    actual_test = test_labels

    # accuracy_train = accuracy_score(actual_train, pred_train)
    # precision_train = precision_score(actual_train, pred_train, average='micro')
    # recall_train = recall_score(actual_train, pred_train, average='micro')
    # f1_train = f1_score(actual_train, pred_train, average='micro')
    #
    # accuracy_test = accuracy_score(actual_test, pred_test)
    # precision_test = precision_score(actual_test, pred_test, average='micro')
    # recall_test = recall_score(actual_test, pred_test, average='micro')
    # f1_test = f1_score(actual_test, pred_test, average='micro')

    accuracy_train, precision_train, recall_train, f1_train = calculate_metrics(pred_train, actual_train)
    accuracy_test, precision_test, recall_test, f1_test = calculate_metrics(pred_test, actual_test)

    print('\nTraining accuracy:\t' + str(accuracy_train))
    print('Training precision:\t' + str(precision_train))
    print('Training recall:\t' + str(recall_train))
    print('Training f1:\t' + str(f1_train))

    print('\nTesting accuracy:\t' + str(accuracy_test))
    print('Testing precision:\t' + str(precision_test))
    print('Testing recall:\t' + str(recall_test))
    print('Testing f1:\t' + str(f1_test))

    if visualize_arg:
        # display test predictions
        display_single_prediction(predictions_test, test_labels, test_images, class_names, 0)
        display_single_prediction2(test_labels,class_names,model,test_images[0])
        display_multiple_prediction(predictions_test, test_labels, test_images, class_names)

        # display train predictions
        display_single_prediction(predictions_train, train_labels, train_images, class_names, 0)
        display_single_prediction2(train_labels,class_names,model,train_images[0])
        display_multiple_prediction(predictions_train, train_labels, train_images, class_names)
