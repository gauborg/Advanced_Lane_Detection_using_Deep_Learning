import cv2
 
# Opens the Video file
fname = "./monument-valley-2.mp4"
video = cv2.VideoCapture(fname)

# image number sequence to start with
i=1000001

while(video.isOpened()):
    ret, frame = video.read()

    try:
        frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
    except cv2.error:
        pass

    # path to write images to ...
    cv2.imwrite('./images/image'+str(i)+'.jpg',frame)
    i+=1
 
video.release()
cv2.destroyAllWindows()