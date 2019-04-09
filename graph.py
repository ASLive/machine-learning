import matplotlib.pyplot as plt
import numpy as np
from hand3d.utils.general import plot_hand

def display(train_images, class_names, train_labels):
    """ display the first 25 images """
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

def display_single_prediction(predictions, test_labels, test_images, class_names, i=0):
    """ prediction next to image """
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)
    plt.show()

def display_single_prediction2(test_labels,class_names,model,img):
    """ labeled x axis with other options """
    img = (np.expand_dims(img,0)) # Add the image to a batch where it's the only member.
    predictions_single = model.predict(img)
    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(24), class_names, rotation=45)
    plt.show()

def display_multiple_prediction(predictions, test_labels, test_images, class_names, num_rows=5, num_cols=3):
    """"Plot the first X test images, their predicted label, and the true label
        Color correct predictions in blue, incorrect predictions in red"""
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

def plot_image(i, predictions_array, true_label, img, class_names):
    """ helper function to plot an image """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plot_hand(img, plt)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    """ helper function to plot values """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(24), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
