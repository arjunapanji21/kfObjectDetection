'''
    File name         : main.py
    Description       : Main file for object tracking
    Author            : Arjuna Panji Prakarsa
    Date created      : 17/01/2021
    Python Version    : 2.7
'''

import cv2
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
from time import sleep
from arjuna import KalmanFilter
from arjuna import Ax12
from arjuna import jarak

def nothing(x):
    #print(x)
    pass

def main():
    #############################################################################################################
    #  MENU AWAL
    #############################################################################################################
    print "[0] Integrated Camera"
    print "[2] Logitech Camera"
    print "[3] Deteksi Dari Video"
    camera = input("Pilih Camera: ")

    if camera == 3:
        camera = 'samples/kalman.mp4'

    print "[1]. Deteksi Menggunakan SIFT"
    print "[2]. Deteksi Menggunakan Kalman Filter"
    print "[3]. Deteksi Menggunakan SIFT & Kalman"
    state = input("Input: ")
    
    servo = False
    servo = raw_input("Turn On Servo(y/n): ")
    if servo == "y":
        servo = True
    else:
        servo = False
        
    debug = input("Debug Mode: ")

    scan_bola = 10
    detek_bola = 1
    ada_bola = False
    area = 0
    wait = 0
    count = 0

    # Call SIFT method (opencv v3.4.2 or lower)
    sift = cv2.xfeatures2d.SIFT_create()

    if servo == True:
        # connecting servo port
        Ax12.open_port()
        Ax12.set_baudrate()

        # create motor object
        servo_kk = Ax12(0) #putar servo ke kiri dan kanan
        servo_ab = Ax12(1) #putar servo ke atas dan bawah

        servo_ab.set_position(512)
        sleep(0.5)
        servo_kk.set_position(512)
        sleep(0.5)

    # batas tengah
    min_kiri = 278
    max_kiri = 229

    min_kanan = 362
    max_kanan = 411

    min_atas = 116
    max_atas = 52

    min_bawah = 244
    max_bawah = 308
    
    window_name = 'Object Detection By Arjuna Panji'
    cv2.namedWindow(window_name)
    cv2.createTrackbar('L_H', window_name, 0, 255, nothing)
    cv2.createTrackbar('L_S', window_name, 0, 255, nothing)
    cv2.createTrackbar('L_V', window_name, 0, 255, nothing)
    cv2.createTrackbar('U_H', window_name, 0, 255, nothing)
    cv2.createTrackbar('U_S', window_name, 0, 255, nothing)
    cv2.createTrackbar('U_V', window_name, 0, 255, nothing)
    cv2.createTrackbar('Threshold', window_name, 0, 255, nothing)

    # bola orange
    cv2.setTrackbarPos('L_H', window_name, 0)
    cv2.setTrackbarPos('L_S', window_name, 200)
    cv2.setTrackbarPos('L_V', window_name, 150)
    cv2.setTrackbarPos('U_H', window_name, 10)
    cv2.setTrackbarPos('U_S', window_name, 255)
    cv2.setTrackbarPos('U_V', window_name, 255)
    cv2.setTrackbarPos('Threshold', window_name, 127)

    # # random test
    # cv2.setTrackbarPos('L_H', window_name, 0)
    # cv2.setTrackbarPos('L_S', window_name, 117)
    # cv2.setTrackbarPos('L_V', window_name, 171)
    # cv2.setTrackbarPos('U_H', window_name, 75)
    # cv2.setTrackbarPos('U_S', window_name, 255)
    # cv2.setTrackbarPos('U_V', window_name, 255)
    # cv2.setTrackbarPos('Threshold', window_name, 127)

    # Create opencv video capture object
    cap = cv2.VideoCapture(camera)

    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    # Get center of the frame
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    cenX_frame = int(cols/2)
    cenY_frame = int(rows/2)

    # Create object for kalman filter
    kfObj = KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)

    while True:
        # Read frame
        _, frame = cap.read()

        font = cv2.FONT_HERSHEY_SIMPLEX
        date = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cv2.putText(frame, date, (10, 20,), font, 0.5, (255,0,0), 2, cv2.LINE_AA)

        l_h = cv2.getTrackbarPos('L_H', window_name)
        l_s = cv2.getTrackbarPos('L_S', window_name)
        l_v = cv2.getTrackbarPos('L_V', window_name)
        u_h = cv2.getTrackbarPos('U_H', window_name)
        u_s = cv2.getTrackbarPos('U_S', window_name)
        u_v = cv2.getTrackbarPos('U_V', window_name)
        th = cv2.getTrackbarPos('Threshold', window_name)

        lowerBall = np.array([l_h, l_s, l_v])
        upperBall = np.array([u_h, u_s, u_v])

        # Convert frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blur the frame
        blur = cv2.medianBlur(hsv, 5)

        # Create a mask from blurred frame
        mask = cv2.inRange(blur, lowerBall, upperBall)

        # Convert to black and white image
        _, thresh = cv2.threshold(mask, th, 255, 0)

        # Refine the image using morphological transformation
        kernal = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernal, iterations=2)

        # Find contours
        _, contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)

            #############################################################################################################
            #  SIFT
            #############################################################################################################
            if area > 1000 and state == 1 or state == "sift":
                (x,y,w,h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, "Object", (x,y), font, 0.5, (0,255,0),2)
                cv2.putText(frame, "X: " + str(x) + "Y: " + str(y), (520, 20), font, 0.5, (0,0,255),2)
                cenX = (x+x+w) / 2
                cenY = (y+y+h) / 2
                roi_color = frame[y:y+h, x:x+w]

                if detek_bola < scan_bola and ada_bola == False :
                    try:
                        wait = 10
                        tmp = "database/tmp" + str(detek_bola)+ ".jpg"
                        cv2.imwrite(tmp, roi_color)
                        img_set = cv2.imread("samples/bola" + str(detek_bola) + ".jpg")
                        img_tmp = cv2.imread("samples/tmp" + str(detek_bola) + ".jpg")
                        #check for similarities between 2 images
                        #kp = key point
                        kp1, desc1 = sift.detectAndCompute(img_set, None)
                        kp2, desc2 = sift.detectAndCompute(img_tmp, None)
                        indexParams = dict(algorithm=0, trees=5)
                        searchParams = dict()
                        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
                        matches = flann.knnMatch(desc1, desc2, k=2)
                        goodPoints = []
                        for m,n in matches:
                            if m.distance < 0.9*n.distance:
                                goodPoints.append(m)

                        num_kp = 0
                        if len(kp1) <= len(kp2):
                            num_kp = len(kp1)
                        else:
                            num_kp = len(kp2)
                            
                        match = float(len(goodPoints)) / float(num_kp) if num_kp != 0 else 0
                        matches_rate = match * 100
                        result = cv2.drawMatches(img_set, kp1, img_tmp, kp2, goodPoints, None)
                        cv2.imshow("result: ", result)
                        if matches_rate > 65 and num_kp > 10:
                            print "ada bola yang mirip data no." + str(detek_bola)
                            ada_bola = True
                        else:
                            detek_bola = detek_bola + 1
                            #print "Objek tidak sama dengan data no." + str(detek_bola) + "..."
                            ada_bola = False
                    except:
                        #print("An exception occurred")
                        pass
                    
                    if debug == 1:
                        print("Detecting = " + str(detek_bola))
                        print("Matches: ", len(goodPoints))
                        print("num_kp= " + str(num_kp))
                        print("Match Rate(%): ", int(matches_rate))

                if ada_bola == True:
                    cv2.putText(frame, "Bola", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                    x,y,w,h = cv2.boundingRect(contour)
                    #roi_hsv = hsv[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
                    #print("Ada Bola")

                    # prediksi jarak
                    r = cenX - x #radius bola
                    prediksi_jarak = jarak(r)
                    cv2.putText(frame, "Jarak: " + str(prediksi_jarak) + " cm", (520, 40), font, 0.5, (255,0,0),2)
                break

            #############################################################################################################
            #  KALMAN FILTER
            #############################################################################################################
            elif area > 1000 and state == 2 or state == "kalman":
                (x,y,w,h) = cv2.boundingRect(contour)
                #cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)
                #cv2.putText(frame, "Object", (x,y), font, 0.5, (0,255,0),2)
                cv2.putText(frame, "X: " + str(x) + "Y: " + str(y), (520, 20), font, 0.5, (0,0,255),2)
                cenX = (x+x+w) / 2
                cenY = (y+y+h) / 2

                predictedCoords = kfObj.Estimate(cenX, cenY)

                # Draw Actual coords from segmentation
                cv2.circle(frame, (int(cenX), int(cenY)), 20, [0,255,0], 2, 8)
                cv2.line(frame,(int(cenX), int(cenY + 20)), (int(cenX + 50), int(cenY + 20)), [0,255,0], 2,8)
                cv2.putText(frame, "Actual", (int(cenX + 50), int(cenY + 20)), font, 0.5, [0,255,0], 2)

                # Draw Kalman Filter Predicted output
                cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [255,0,0], 2, 8)
                cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [255, 0, 0], 2, 8)
                cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)), font, 0.5, [255, 0, 0], 2)

                ada_bola = True
                wait = 10

                # prediksi jarak
                r = cenX - x #radius bola
                prediksi_jarak = jarak(r)
                cv2.putText(frame, "Jarak: " + str(prediksi_jarak) + " cm", (520, 40), font, 0.5, (255,0,0),2)
                break
            
            #############################################################################################################
            #  SIFT + KALMAN
            #############################################################################################################
            elif area > 1000 and state == 3 or state == "sift+kalman":
                (x,y,w,h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, "Object", (x,y), font, 0.5, (0,255,0),2)
                cv2.putText(frame, "X: " + str(x) + "Y: " + str(y), (520, 20), font, 0.5, (0,0,255),2)
                cenX = (x+x+w) / 2
                cenY = (y+y+h) / 2
                roi_color = frame[y:y+h, x:x+w]
                if (wait < 0 and (len(contour)) > 1):
                    wait = 10
                print (wait)

                if detek_bola < scan_bola and ada_bola == False and wait == 0:
                    try:
                        wait = 10
                        tmp = "database/tmp" + str(detek_bola)+ ".jpg"
                        cv2.imwrite(tmp, roi_color)
                        img_set = cv2.imread("database/bola" + str(detek_bola) + ".jpg")
                        img_tmp = cv2.imread("database/tmp" + str(detek_bola) + ".jpg")
                        #check for similarities between 2 images
                        #kp = key point
                        kp1, desc1 = sift.detectAndCompute(img_set, None)
                        kp2, desc2 = sift.detectAndCompute(img_tmp, None)
                        indexParams = dict(algorithm=0, trees=5)
                        searchParams = dict()
                        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
                        matches = flann.knnMatch(desc1, desc2, k=2)
                        goodPoints = []
                        for m,n in matches:
                            if m.distance < 0.9*n.distance:
                                goodPoints.append(m)

                        num_kp = 0
                        if len(kp1) <= len(kp2):
                            num_kp = len(kp1)
                        else:
                            num_kp = len(kp2)
                            
                        match = float(len(goodPoints)) / float(num_kp) if num_kp != 0 else 0
                        matches_rate = match * 100
                        result = cv2.drawMatches(img_set, kp1, img_tmp, kp2, goodPoints, None)
                        cv2.imshow("result: ", result)
                        if matches_rate > 65 and num_kp > 10:
                            print "ada bola yang mirip data no." + str(detek_bola)
                            ada_bola = True
                        else:
                            detek_bola = detek_bola + 1
                            #print "Objek tidak sama dengan data no." + str(detek_bola) + "..."
                            ada_bola = False
                    except:
                        print("An exception occurred")
                    
                    if debug == 1:
                        print("Detecting = " + str(detek_bola))
                        print("Matches: ", len(goodPoints))
                        print("num_kp= " + str(num_kp))
                        print("Match Rate(%): ", int(matches_rate))

                if ada_bola == True:
                    #cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)
                    #cv2.putText(frame, "Object", (x,y), font, 0.5, (0,255,0),2)
                    cv2.putText(frame, "X: " + str(x) + "Y: " + str(y), (520, 20), font, 0.5, (0,0,255),2)
                    cenX = (x+x+w) / 2
                    cenY = (y+y+h) / 2

                    predictedCoords = kfObj.Estimate(cenX, cenY)

                    # Draw Actual coords from segmentation
                    cv2.circle(frame, (int(cenX), int(cenY)), 20, [0,255,0], 2, 8)
                    cv2.line(frame,(int(cenX), int(cenY + 20)), (int(cenX + 50), int(cenY + 20)), [0,255,0], 2,8)
                    cv2.putText(frame, "Actual", (int(cenX + 50), int(cenY + 20)), font, 0.5, [0,255,0], 2)

                    # Draw Kalman Filter Predicted output
                    cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [255,0,0], 2, 8)
                    cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [255, 0, 0], 2, 8)
                    cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)), font, 0.5, [255, 0, 0], 2)

                    ada_bola = True
                    wait = 10

                    # prediksi jarak
                    r = cenX - x #radius bola
                    prediksi_jarak = jarak(r)
                    cv2.putText(frame, "Jarak: " + str(prediksi_jarak) + " cm", (520, 40), font, 0.5, (255,0,0),2)
                break

        #############################################################################################################
        #  Servo
        #############################################################################################################
        if servo == True:
            if ada_bola == True and state == 1 or state == "sift":
                # kiri dikit
                if cenX > max_kiri and cenX < min_kiri:
                    #print "kiri dikit"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos + 7)
                    if pos >= 1000:
                        servo_kk.set_position(1000)

                # kiri banyak
                elif cenX < max_kiri:
                    #print "kiri banyak"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos + 23)
                    if pos >= 1000:
                        servo_kk.set_position(1000)
                else:
                    #print "tengah"
                    pass

                # kanan dikit
                if cenX > min_kanan and cenX < max_kanan:
                    #print "kanan dikit"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos - 7)
                    if pos <= 23:
                        servo_kk.set_position(23)

                # kanan banyak
                elif cenX > max_kanan:
                    #print "kanan banyak"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos - 23)
                    if pos <= 23:
                        servo_kk.set_position(23)
                else:
                    #print "tengah"
                    pass

                # atas dikit
                if cenY > max_atas and cenY < min_atas:
                    #print "atas dikit"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos + 3)
                    if pos >= 1000:
                        servo_ab.set_position(1000)

                # atas banyak
                elif cenY < max_atas:
                    #print "atas banyak"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos + 12)
                    if pos >= 1000:
                        servo_ab.set_position(1000)
                else:
                    #print "tengah"
                    pass

                # bawah dikit
                if cenY > min_bawah and cenY < max_bawah:
                    #print "bawah dikit"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos - 3)
                    if pos <= 23:
                        servo_ab.set_position(23)

                # bawah banyak
                elif cenY > max_bawah:
                    #print "bawah banyak"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos - 12)
                    if pos <= 23:
                        servo_ab.set_position(23)
                else:
                    #print "tengah"
                    pass
            
            elif ada_bola == True and state == 2 or state == "kalman":
                # kiri dikit
                if predictedCoords[0] > max_kiri and predictedCoords[0] < min_kiri:
                    #print "kiri dikit"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos + 7)
                    if pos >= 1000:
                        servo_kk.set_position(1000)

                # kiri banyak
                elif predictedCoords[0] < max_kiri:
                    #print "kiri banyak"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos + 23)
                    if pos >= 1000:
                        servo_kk.set_position(1000)
                else:
                    #print "tengah"
                    pass

                # kanan dikit
                if predictedCoords[0] > min_kanan and predictedCoords[0] < max_kanan:
                    #print "kanan dikit"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos - 7)
                    if pos <= 23:
                        servo_kk.set_position(23)

                # kanan banyak
                elif predictedCoords[0] > max_kanan:
                    #print "kanan banyak"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos - 23)
                    if pos <= 23:
                        servo_kk.set_position(23)
                else:
                    #print "tengah"
                    pass

                # atas dikit
                if predictedCoords[1] > max_atas and predictedCoords[1] < min_atas:
                    #print "atas dikit"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos + 3)
                    if pos >= 1000:
                        servo_ab.set_position(1000)

                # atas banyak
                elif predictedCoords[1] < max_atas:
                    #print "atas banyak"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos + 12)
                    if pos >= 1000:
                        servo_ab.set_position(1000)
                else:
                    #print "tengah"
                    pass

                # bawah dikit
                if predictedCoords[1] > min_bawah and predictedCoords[1] < max_bawah:
                    #print "bawah dikit"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos - 3)
                    if pos <= 23:
                        servo_ab.set_position(23)

                # bawah banyak
                elif predictedCoords[1] > max_bawah:
                    #print "bawah banyak"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos - 12)
                    if pos <= 23:
                        servo_ab.set_position(23)
                else:
                    #print "tengah"
                    pass

            elif ada_bola == True and state == 3 or state == "sift+kalman":
                # kiri dikit
                if predictedCoords[0] > max_kiri and predictedCoords[0] < min_kiri:
                    #print "kiri dikit"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos + 7)
                    if pos >= 1000:
                        servo_kk.set_position(1000)

                # kiri banyak
                elif predictedCoords[0] < max_kiri:
                    #print "kiri banyak"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos + 23)
                    if pos >= 1000:
                        servo_kk.set_position(1000)
                else:
                    #print "tengah"
                    pass

                # kanan dikit
                if predictedCoords[0] > min_kanan and predictedCoords[0] < max_kanan:
                    #print "kanan dikit"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos - 7)
                    if pos <= 23:
                        servo_kk.set_position(23)

                # kanan banyak
                elif predictedCoords[0] > max_kanan:
                    #print "kanan banyak"
                    pos = servo_kk.get_position()
                    servo_kk.set_position(pos - 23)
                    if pos <= 23:
                        servo_kk.set_position(23)
                else:
                    #print "tengah"
                    pass

                # atas dikit
                if predictedCoords[1] > max_atas and predictedCoords[1] < min_atas:
                    #print "atas dikit"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos + 3)
                    if pos >= 1000:
                        servo_ab.set_position(1000)

                # atas banyak
                elif predictedCoords[1] < max_atas:
                    #print "atas banyak"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos + 12)
                    if pos >= 1000:
                        servo_ab.set_position(1000)
                else:
                    #print "tengah"
                    pass

                # bawah dikit
                if predictedCoords[1] > min_bawah and predictedCoords[1] < max_bawah:
                    #print "bawah dikit"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos - 3)
                    if pos <= 23:
                        servo_ab.set_position(23)

                # bawah banyak
                elif predictedCoords[1] > max_bawah:
                    #print "bawah banyak"
                    pos = servo_ab.get_position()
                    servo_ab.set_position(pos - 12)
                    if pos <= 23:
                        servo_ab.set_position(23)
                else:
                    #print "tengah"
                    pass

            # # batas dalam
            # cv2.rectangle(frame, (cenX_frame - (int(cols/15)),cenY_frame - (int(rows/10))),(cenX_frame + (int(cols/15)), cenY_frame + (int(rows/10))), (0,255,0), 2)
            # cv2.putText(frame, str(cenX_frame - (int(cols/15))), (cenX_frame - (int(cols/15)), cenY_frame - (int(rows/10))), font, 0.5, [255, 0, 0], 2)
            # cv2.putText(frame, str(cenX_frame + (int(cols/15))), (cenX_frame + (int(cols/15)), cenY_frame + (int(rows/10))), font, 0.5, [255, 0, 0], 2)

            # cv2.putText(frame, str(cenY_frame - (int(cols/10))), (cenX_frame + (int(cols/15)), cenY_frame - (int(rows/10))), font, 0.5, [0, 0, 255], 2)
            # cv2.putText(frame, str(cenY_frame + (int(cols/10))), (cenX_frame - (int(cols/15)), cenY_frame + (int(rows/10))), font, 0.5, [0, 0, 255], 2)

            # # batas luar
            # cv2.rectangle(frame, (cenX_frame - (int(cols/7)),cenY_frame - (int(rows/5))),(cenX_frame + (int(cols/7)), cenY_frame + (int(rows/5))), (0,255,0), 2)
            # cv2.putText(frame, str(cenX_frame - (int(cols/7))), (cenX_frame - (int(cols/7)), cenY_frame - (int(rows/5))), font, 0.5, [255, 0, 0], 2)
            # cv2.putText(frame, str(cenX_frame + (int(cols/7))), (cenX_frame + (int(cols/7)), cenY_frame + (int(rows/5))), font, 0.5, [255, 0, 0], 2)

            # cv2.putText(frame, str(cenY_frame - (int(cols/5))), (cenX_frame + (int(cols/7)), cenY_frame - (int(rows/5))), font, 0.5, [0, 0, 255], 2)
            # cv2.putText(frame, str(cenY_frame + (int(cols/5))), (cenX_frame - (int(cols/7)), cenY_frame + (int(rows/5))), font, 0.5, [0, 0, 255], 2)

        #############################################################################################################
        #  Logika
        #############################################################################################################
        # print(str(cenX))
        # print(str(cenY))
        # print(str(int(predictedCoords[0])))
        # print(str(int(predictedCoords[1])))

        if wait > 0:
            wait = wait - 1

        else:
            ada_bola = False
            detek_bola = 1
            #print("stop")
            #print "Bola Tidak Ada"

        if ada_bola == True:
            count = 0.0
        elif ada_bola == False:
            count += 0.2

        if camera == 0 or camera == 2:
            cv2.imshow(window_name, morph)
            cv2.imshow("Frame", frame)

        else:
            cv2.imshow(window_name, frame)
            cv2.waitKey(50)

        if debug == 1:
            print("area= "+str(area))
            print("state= "+str(state))
            print("ada bola= "+str(ada_bola))
            print("detek bola= "+str(detek_bola))
            print("wait= "+str(wait))

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break 

if __name__ == "__main__":
    # Execute main
    main()