import os
import cv2

if not os.path.exists("videodata"):
    os.makedirs("videodata")

if not os.path.exists("videodata/train"):
    os.makedirs("videodata/train")
if not os.path.exists("videodata/test"):
    os.makedirs("videodata/test")

if not os.path.exists("videodata/train/hello"):
    os.makedirs("videodata/train/hello")
if not os.path.exists("videodata/train/thankyou"):
    os.makedirs("videodata/train/thankyou")

cap=cv2.VideoCapture(0)
directory="videodata/train"
for direct in os.listdir(directory):
    while True:
        count={
            'hello':len(os.listdir(directory+"/hello")),
            'thankyou':len(os.listdir(directory+"/thankyou"))
        }
        result = cv2.VideoWriter(directory+"/"+direct+"/"+str(count[direct])+".mp4", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
        while (cap.isOpened()):
            ret,img=cap.read()
            if ret==True:
                result.write(img)
                cv2.imshow('Frame', img)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
            else:
                break
        i=int(input("enter 1 for continue else 0:"))
        if i:
            continue
        else:
            break    
    result.release()
cap.release()
cv2.destroyAllWindows()

