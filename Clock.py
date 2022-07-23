import cv2
import math
import numpy as np
img = cv2.imread(r"/home/shyam/Documents/Alarm clock/Alarm_clock/clock(1).jpg")
img = cv2.resize(img,None,fx = 0.2,fy = 0.2)

h,w,d = img.shape

size = 250
x0 = int(w/2-size/2)
y0 = int(h/2-size/2)
x1 = int(w/2+size/2)
y1 = int(h/2+size/2)
#print((x0,y0),(x1,y1))
blank = np.zeros(img.shape,np.uint8)

blank = cv2.rectangle(blank,(x0,y0),(x1,y1),(255,255,255),-1)
blank = cv2.bitwise_and(img,blank)
img = blank
blank = np.zeros(img.shape,np.uint8)


img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower = np.array([37,0,0])
upper = np.array([99,255,255])
mask = cv2.inRange(img_hsv,lower,upper)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#bitand = cv2.bitwise_or(img,img,mask=mask)
#bitand = img+mask
#mask = cv2.bitwise_not(mask)


#mask = cv2.Canny(mask,10,255)
contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

blank = np.zeros(mask.shape,np.uint8)
img_copy = img.copy()
approx_contour = []
time = []
for i in range(0,len(contour)):
    epsilon = cv2.arcLength(contour[i],True) * 0.01
    approx = cv2.approxPolyDP(contour[i],epsilon,True)
    approx_contour.append(approx)
    #print(cv2.contourArea(approx))
    cv2.drawContours(img_copy,[approx],0,(0,255,0),-1)
for i in range(0,len(contour)):
    rows,cols = img.shape[:2]
    
    [vx,vy,x,y] = cv2.fitLine(approx_contour[i], cv2.DIST_L2,0,0.01,0.01)
    deg = math.atan(vy/vx)
    deg = deg+(3/2)*math.pi
    print(deg)
    time.append(deg)
    #print(approx)
    #print([vx,vy,x,y])
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    #cv2.line(img_copy,(cols-1,righty),(0,lefty),(0,255,0),2)
    #cv2.imshow("blank",img_copy)
    #cv2.waitKey(0)
print("hour: ",int(time[1]*12/(2*math.pi)))
print("minute: ",int(time[0]*60/(2*math.pi)))

cv2.imshow("segregate",img_copy)
cv2.imwrite("segregate.jpg",img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
