# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:27:53 2019

@author: Carter
"""
import numpy as np
import matplotlib.pyplot as plt

#ploting accuracies
def plot_acc(cm, title="Accuracy"):
    rand = [1/4,1/2,1/3]
    cm = np.array([cm,rand])
    print(cm)
    accs = ["Our Accuracy", "Random Acc"]#np.arange(cm.shape[0])
    classes = ["Bolts", "Big Gear", "Small Gear"]
    cmap = plt.cm.Blues
    fig, ax = plt.subplots()
    normalize = False
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=accs,
           title=title,
           ylabel='Accuracy',
           xlabel='Class')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    
def plot_confusion(cm, title="Confusion Matrix"):
        classes = np.arange(cm.shape[0])
        cmap = plt.cm.Blues
        fig, ax = plt.subplots()
        normalize = False
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()
   