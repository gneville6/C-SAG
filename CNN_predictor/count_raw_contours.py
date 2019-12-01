from color_threshold_attempt_wrapped import ColorThresholdAttempt
import os
import cv2


def test_function():
    path = '/home/sean/Images/Test_set/3bolts1big0small/349.jpeg'
    ColorObj = ColorThresholdAttempt()
    im = cv2.imread(path)
    screw_num = ColorObj.count_number_of_screws(im)
    print(screw_num)
    # ColorObj.preview(im1)


def run_all_count_screws():
    ColorObj = ColorThresholdAttempt()

    rawImageFolderPath = '/home/sean/Images/Test_set'

    for j in os.scandir(rawImageFolderPath):
        path1 = j.path
        print(path1)

        count_dict = {}

        for f in os.scandir(path1):
            if not f.is_file():
                continue

            im = cv2.imread(f.path)
            # print(f.path)
            im = ColorObj.resize_image(im, 1.00)

            # ColorObj.preview(im)
            num_screws = ColorObj.count_number_of_screws(im)

            if num_screws in count_dict:
                count_dict[num_screws] = count_dict[num_screws] + 1
            else:
                count_dict[num_screws] = 1

        
        print(count_dict)

            # im1 = ColorObj.process_image_just_crop(im)

            # ColorObj.preview(im1)



if __name__ == "__main__":
    # test_function()
    run_all_count_screws()