from color_threshold_attempt_wrapped import ColorThresholdAttempt
import os
import cv2

def test_function():
    path = '/home/sean/Images/Test_set/3bolts1big0small/349.jpeg'





if __name__ == "__main__":
    ColorObj = ColorThresholdAttempt()

    rawImageFolderPath = '/home/sean/Images/Test_set'

    for j in os.scandir(rawImageFolderPath):
        path1 = j.path
        print(path1)
        for f in os.scandir(path1):
            if not f.is_file():
                continue

            im = cv2.imread(f.path)
            print(f.path)
            im = ColorObj.resize_image(im, 1.00)

            im1 = ColorObj.count_number_of_screws(im)

            # im1 = ColorObj.process_image_just_crop(im)



            # ColorObj.preview(im1)