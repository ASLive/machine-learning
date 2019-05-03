import numpy as np
from graph import *
from ml import *
from data import *
import sys
from sklearn.metrics import *

print("Tensorflow version "+tf.__version__)

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

    images, labels, class_names = load_data()
    print(len(images))
    print(len(images[0]))
    print(len(images[0][0]))
    #train_images, train_labels, test_images, test_labels, class_names = load_data()
    model = ml_model(images, labels, retrain_arg, no_save_arg)

    #test_loss, test_acc = model.evaluate(test_images, test_labels)
    train_loss, train_acc = model.evaluate(images, labels)

    #predictions_test = model.predict(test_images)
    predictions_train = model.predict(images)

    pred_train = np.argmax(predictions_train, axis=1)
    #pred_test = np.argmax(predictions_test, axis=1)
    actual_train = labels
    #actual_test = test_labels

    cm_train = confusion_matrix(y_true=actual_train, y_pred=pred_train)
    #cm_test = confusion_matrix(y_true=actual_test, y_pred=pred_test)

    accuracy_train = accuracy_score(actual_train, pred_train)
    precision_train = precision_score(actual_train, pred_train, average='micro')
    recall_train = recall_score(actual_train, pred_train, average='micro')
    f1_train = f1_score(actual_train, pred_train, average='micro')

    #accuracy_test = accuracy_score(actual_test, pred_test)
    #precision_test = precision_score(actual_test, pred_test, average='micro')
    #recall_test = recall_score(actual_test, pred_test, average='micro')
    #f1_test = f1_score(actual_test, pred_test, average='micro')

    print('\nTraining accuracy:\t' + str(accuracy_train))
    print('Training precision:\t' + str(precision_train))
    print('Training recall:\t' + str(recall_train))
    print('Training f1:\t' + str(f1_train))

    #print('\nTesting accuracy:\t' + str(accuracy_test))
    #print('Testing precision:\t' + str(precision_test))
    #print('Testing recall:\t' + str(recall_test))
    #print('Testing f1:\t' + str(f1_test))

    print('\nTraining Confusion Matrix:\n' + str(cm_train))
    #print('\nTesting Confusion Matrix:\n' + str(cm_test))

    if visualize_arg:
        # display test predictions
        #display_single_prediction(predictions_test, test_labels, test_images, class_names, 0)
        #display_single_prediction2(test_labels,class_names,model,test_images[0])
        #display_multiple_prediction(predictions_test, test_labels, test_images, class_names)

        # display train predictions
        display_single_prediction(predictions_train, labels, images, class_names, 0)
        display_single_prediction2(labels,class_names,model,images[0])
        display_multiple_prediction(predictions_train, labels, images, class_names)
