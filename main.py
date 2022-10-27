import cv2
import detectAgecopy as d
import PeopleTrackcopy as p
import imutils

RUN  = True
vid = cv2.VideoCapture(0)

ret, cache = vid.read()
while RUN:

    ret, frame = vid.read()
    frame = imutils.resize(frame, width=600)
    cache =  frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       
    d.I(frame)   
    p.main(frame)
    # text,x1,y1,x2,y2 = p.main(frame)
    # text1 = ""
    # if "none" in text:
    #     print("yes")
    #     cv2.destroyAllWindows()
    #     cv2.imshow("Not detected", cache)
    # else:
    #     if "person" in text:
    #         text1 = d.I(frame)
    #     cv2.destroyAllWindows()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.putText(frame, f"{text}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,2,255), 2, cv2.LINE_AA)
        # cv2.putText(frame, f"{text1}", (5,55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,2,255), 2, cv2.LINE_AA)
    cv2.imshow("Detecting", frame)

    
vid.release()
cv2.destroyAllWindows()