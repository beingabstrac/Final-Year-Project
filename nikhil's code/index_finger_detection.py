import cv2
import numpy as np
import math


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/Users/RishiDeep/#rD/Final Year Project/Untitled.mov')
while(cap.isOpened()):
    # read image
    ret, img = cap.read()


    # get hand data from the rectangle sub window on the screen
    sx1,sy1,sx2,sy2 = 550,550,0,0 
    cv2.rectangle(img, (sx1,sy1), (sx2,sy2), (0,255,0),0)
    crop_img = img[sx2:sx1, sy2:sy1]

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # show thresholded image
    cv2.imshow('Thresholded', thresh1)

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    #drawing fit ellipse
    #ellipse = cv2.fitEllipse(cnt)
    #cv2.ellipse(drawing, ellipse ,(255, 0, 0), 0)
    #cv2.circle(drawing, tuple([int(ellipse[0][0]),int(ellipse[0][1])]), 1, (255,255,255), 1)
    #cv2.circle(drawing, tuple([int(ellipse[1][0]),int(ellipse[1][1])]), 1, (255,255,255), 1)

    #creating new frame to show index fingure
    index = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(index, [cnt], 0, (0, 255, 0), 0)

    #center of mass
    M = cv2.moments(cnt)
    mass_x = M['m10']/M['m00']
    mass_y = M['m01']/M['m00']
    cv2.circle(drawing, tuple([int(mass_x),int(mass_y)]), 1, (255,255,255), 1)
    cv2.circle(index, tuple([int(mass_x),int(mass_y)]), 1, (255,255,255), 1)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    # finding convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all defects (between fingers)
    # with angle > 90 degrees and ignore defects
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])


        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 2, [0,0,255], -1)
            cv2.circle(drawing, far, 3, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        cv2.line(crop_img,start, end, [0,255,0], 2)

    # define actions required
    if count_defects == 1:

        cv2.putText(img,"ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        circle = cv2.minEnclosingCircle(cnt)

        x = circle[0][0]
        y = circle[0][1]
        r = circle[1]

        cv2.circle(drawing, tuple([int(x),int(y)]), int(r), (255,255,0), 1)
        cv2.circle(drawing, tuple([int(x),int(y)]), 1, (255,255,0), 1)

        # determine the most extreme points along the contour
        point = [-1,-1,-1,-1]

        point[0] = tuple(cnt[cnt[:, :, 0].argmin()][0])          #left
        point[1] = tuple(cnt[cnt[:, :, 0].argmax()][0])          #right
        point[2] = tuple(cnt[cnt[:, :, 1].argmin()][0])          #top
        point[3] = tuple(cnt[cnt[:, :, 1].argmax()][0])          #bottom

        cv2.circle(drawing, point[0], 4, (0, 0, 255), -1)
        cv2.circle(drawing, point[1], 4, (0, 255, 0), -1)
        cv2.circle(drawing, point[2], 4, (255, 0, 0), -1)
        cv2.circle(drawing, point[3], 4, (255, 255, 0), -1) 

        
        def near(point , border):
            if(point >= border-5 and point <= border+5):
                return True
            return False
            
        

        m = []
        for i in range(4):
            #remove if the point is near the bountry (optimization)
            if( near(point[i][0],0 ) or near(point[i][0],sx1-sx2) or near(point[i][1],0) or near(point[i][1],sy1-sy2) ):
                dist = -1

            #distance from center of mass
            else:
                dist = math.sqrt((point[i][0] - mass_x)**2 + (point[i][1] - mass_y)**2)
            m.append(tuple([dist, i]))

        m.sort(key=lambda x: x[0])
     
        cv2.circle(index, point[m[3][1]], 4, (0, 0, 255), -1)
 





    elif count_defects == 2:
        cv2.putText(img, "TWO", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    elif count_defects == 3:
        cv2.putText(img,"THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img,"FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # show appropriate images in windows
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img, index))
    cv2.imshow('Contours', all_img)

    k = cv2.waitKey(10)
    if k == 27:
        break
