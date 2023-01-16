import numpy as np
import cv2
import os
import glob
import time


confidenceThreshold = 0.4
NMSThreshold = 0.3

modelConfiguration = r"C:\Users\MSI 1\OneDrive\Desktop\yolo\training\yolov4-custom.cfg"
modelWeights = r"C:\Users\MSI 1\OneDrive\Desktop\yolo\training\yolov4-custom_last.weights"

labelsPath = r"C:\Users\MSI 1\OneDrive\Desktop\yolo\training\obj.names"
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


path = r"C:\Users\MSI 1\OneDrive\Desktop\yolo\augmented_stacked_images\*.bmp"
for file in glob.glob(path):
    image = cv2.imread(file)
    c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start = time.time()


    



    #image = cv2.imread(r'C:\Users\MSI 1\OneDrive\Desktop\testing_images\testing_cropped_image\16.bmp')
    #image = cv2.resize(image, (720,640))
    (H, W) = image.shape[:2]

    #Determine output layer names
    layerName = net.getUnconnectedOutLayersNames() 
    #layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    #print(len(detectionNMS))

    if(len(detectionNMS) > 0  ):
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (ptx,pty) = (x+w/2,y+h/2)
            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (ptx, pty), (ptx + w, pty + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            confidence= round(confidences[1],2)
##            print("x",x)
##            print("y", y)
##            print("w", w)
            #print(confidence)
            ptx,pty=x+(w/2),y+(w/2)
            #print("ptx", ptx, "pty", pty)
            
 
            #print(text)

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    



            #wprint(confidences)
##    else:
##        cv2.putText(image, "Screw Missing", (525, 525), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2 )

        

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    end = time.time()
    print("Execution time of the program is -", end-start)
    cv2.waitKey(1000)

    #main()
    #print("___ &s second ___" % (time.time()- start_time))




