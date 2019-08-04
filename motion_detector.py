import cv2, time, pandas
import os, os.path
from pygame import mixer
from datetime import datetime
import easygui, zipfile

first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])
filename = 'video.avi'  #for video output
frames_per_second = 24.0 #for video
res = '480p' #for video
count=1

video=cv2.VideoCapture(0)

#codes for saving video
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]

    change_res(cap, width, height)
    return width, height

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(video, res))

##########for  video save close
while True:
    check, frame = video.read()
    out.write(frame)
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1

        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())#thisisthe timestamp function of the python
        mixer.init()#sound
        mixer.music.load("a.mp3")#sound
        mixer.music.play()#sound
        #######################print photo###################

        # save_path='C:\Users\DELL\Downloads\Compressed\webcam_motion_detection-master\res'
        filname=str(count)+"motion.jpg"
        count += 1
        cv2.imwrite(filename=filname, img=frame)
        img_new = cv2.imread(filname, cv2.IMREAD_GRAYSCALE)
        img_new = cv2.imshow("Captured Image", img_new)

        ######################print photo##############################

    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())


    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)


    key=cv2.waitKey(1)

    if key==ord('q'):
        # easygui.msgbox("Please buy pro version", title="Webcam Motion Detector")## this is prompt for pro version

        if status==1:
            times.append(datetime.now())#used timestamp [not in db]
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")#saves time data in .csv format


video.release()
out.release()
cv2.destroyAllWindows()

