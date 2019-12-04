from color_threshold_attempt_wrapped import ColorThresholdAttempt
import os
import cv2
import functools
import matplotlib as plt
import numpy as np

def plot_confusion(self, cm, title="Confusion Matrix"):
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


def test_function():
    path = '/home/sean/Images/Test_set/3bolts1big0small/349.jpeg'
    ColorObj = ColorThresholdAttempt()
    im = cv2.imread(path)
    screw_num = ColorObj.count_number_of_screws(im)
    print(screw_num)
    # ColorObj.preview(im1)

def get_total_number_screws(input_string):
    '''given folder name return total number of screws '''

    # filter string to only contain numbers
    res = ''.join(filter(lambda x: x.isdigit(), input_string))

    # # sum digits in string
    # total_num = functools.reduce(lambda a,b: int(a) + int(b), res)

    grey_bolts = int(res[0])
    red_bolt = int(res[1])
    blue_bolts = int(res[2])

    return grey_bolts, red_bolt, blue_bolts
    
def run_all_count_screws():
    ColorObj = ColorThresholdAttempt()

    rawImageFolderPath = '/home/sean/Images/Test_set'

    confusion_red_dict = {}
    red_correct = 0
    red_incorrect = 0

    blue_correct = 0
    blue_incorrect = 0

    grey_correct = 0
    grey_incorrect = 0

    for j in os.scandir(rawImageFolderPath):
        path1 = j.path
        # print(path1)

        # count_dict = {}
        folder_name = j.name
        grey_bolts, red_bolt, blue_bolts = get_total_number_screws(folder_name)
        print(folder_name)



        for f in os.scandir(path1):
            if not f.is_file():
                continue

            im = cv2.imread(f.path)
            # print(f.path)
            im = ColorObj.resize_image(im, 1.00)

            # ColorObj.preview(im)
            # num_screws = ColorObj.count_number_of_screws(im)

            red_predict, blue_predict, grey_predict = ColorObj.classify_image_contour(im)

            if red_bolt == red_predict:
                red_correct += 1
            else:
                red_incorrect += 1

            if blue_bolts == blue_predict:
                blue_correct += 1
            else:
                blue_incorrect += 1

            if grey_bolts == grey_predict:
                grey_correct += 1
            else:
                grey_incorrect += 1

        # print(confusion_dict)
    # return confusion_dict
    print("red: ", red_correct, red_incorrect, red_correct / (red_correct + red_incorrect))
    print("blue: ", blue_correct, blue_incorrect, blue_correct / (blue_correct + blue_incorrect))
    print("grey: ", grey_correct, grey_incorrect, grey_correct / (grey_correct + grey_incorrect))
        # print(count_dict)

            # im1 = ColorObj.process_image_just_crop(im)

            # ColorObj.preview(im1)



if __name__ == "__main__":
    # test_function()
    run_all_count_screws()