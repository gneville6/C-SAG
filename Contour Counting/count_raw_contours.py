from color_threshold_attempt_wrapped import ColorThresholdAttempt
import os
import cv2
import functools

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

    # sum digits in string
    total_num = functools.reduce(lambda a,b: int(a) + int(b), res)

    return total_num
    
def run_all_count_screws():
    ColorObj = ColorThresholdAttempt()

    rawImageFolderPath = '/home/sean/Images/Test_set'

    confusion_dict = {}

    for j in os.scandir(rawImageFolderPath):
        path1 = j.path
        # print(path1)

        count_dict = {}
        folder_name = j.name
        real_num_screws = get_total_number_screws(folder_name)
        print(folder_name)

        count_correct = 0
        count_incorrect = 0

        for f in os.scandir(path1):
            if not f.is_file():
                continue

            im = cv2.imread(f.path)
            # print(f.path)
            im = ColorObj.resize_image(im, 1.00)

            # ColorObj.preview(im)
            # num_screws = ColorObj.count_number_of_screws(im)

            num_screws, blue_screws, grey_screws = ColorObj.classify_image_contour(im)

            if num_screws in count_dict:
                count_dict[num_screws] = count_dict[num_screws] + 1
            else:
                count_dict[num_screws] = 1

            if num_screws == real_num_screws:
                count_correct += 1
            else:
                count_incorrect += 1

        confusion_dict[folder_name] = [count_correct, count_incorrect]
        accuracy = (count_correct/(count_correct + count_incorrect)) * 100
        accuracy = str(round(accuracy, 2))
        print(accuracy)

        file = open("save_data_contour.txt","a+")
        file.write(folder_name+","+str(accuracy)+","+str(count_correct)+","+str(count_incorrect)+'\n')
        file.close()
        # print(confusion_dict)
    return confusion_dict
        # print(count_dict)

            # im1 = ColorObj.process_image_just_crop(im)

            # ColorObj.preview(im1)



if __name__ == "__main__":
    # test_function()
    run_all_count_screws()