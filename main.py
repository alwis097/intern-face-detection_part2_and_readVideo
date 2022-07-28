import cv2 

# x= "C:\Users\DELL\OneDrive\Desktop\Intern\Week8\Face Detection Web Apps\Face Detection Web Apps\static\uploads\Pubg Lobby Screen Live Wallpaper Free.mp4"
video=cv2.VideoCapture("C:/Users/DELL/OneDrive/Desktop/Intern/Week8/Face Detection Web Apps/Face Detection Web Apps/static/1.mp4",0)

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame=video.read()
    faces=faceDetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        x1,y1=x+w, y+h
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 1)
        cv2.line(frame, (x,y), (x+30, y),(255,0,255), 6) #Top Left
        cv2.line(frame, (x,y), (x, y+30),(255,0,255), 6)

        cv2.line(frame, (x1,y), (x1-30, y),(255,0,255), 6) #Top Right
        cv2.line(frame, (x1,y), (x1, y+30),(255,0,255), 6)

        cv2.line(frame, (x,y1), (x+30, y1),(255,0,255), 6) #Bottom Left
        cv2.line(frame, (x,y1), (x, y1-30),(255,0,255), 6)

        cv2.line(frame, (x1,y1), (x1-30, y1),(255,0,255), 6) #Bottom right
        cv2.line(frame, (x1,y1), (x1, y1-30),(255,0,255), 6)

    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()