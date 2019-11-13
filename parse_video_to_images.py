import cv2

vidcap = cv2.VideoCapture('4bolts2big2small.mp4')
success,image = vidcap.read()
count = 0

while success:
    success, image = vidcap.read()
    if count % 30 == 0:
        cv2.imwrite("%d.jpg" % count, image)
    count += 1