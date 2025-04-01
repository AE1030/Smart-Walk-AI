#Import Statements 
import cv2 
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import math as m
import pyttsx3
from collections import deque
from num2words import num2words

#Setup the model and the camera
model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite')
cam = cv2.VideoCapture(0)

#setup the voice engine
engine = pyttsx3.init()
engine.setProperty("rate",140)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[2].id)
engine.say("")
engine.say("Hi, how are you doing")
engine.runAndWait()

#coco list
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

#list of objects we will be detecting from coco
my_file = open("objectsofinterest.txt", "r")
data = my_file.read()
objects_of_interest = data.split("\n")

#List of approximate width for each of the objects we will be detecting
my_file = open("Dimensions.txt")
data = my_file.read()
dimensions_list = data.split("\n")

#Data for loop logic
frame_count = 0
start_time = time.time()
max_det = 1
prev_object = {}
confirmation_deq = deque(maxlen=5)
cooldown_time = 25 #Dont detect the same object within 25 seconds
image_width = 240
bb_threshold = 0.7*image_width
center_of_frame = image_width / 2

#This function aims to confirm an object for atleast 3/5 frames before making an announcement which limits flickering/hollusinations by our model
def confirm_object(name):
    confirmation_deq.append(name)
    
    if name is not None:
        count = confirmation_deq.count(name)
        stability = count>=2
    
    else:
        stability = False
        
    return stability
    
#Thus function aims to get the approximate distance of each object
def get_distance(bb_width, x_origin, index):
    fx = 500  #Focal length for typical webcams
    cx = center_of_frame
    #theta = m.atan(x_origin-cx/fx)
    
    if dimensions_list[index] != "None":
        distance = abs((int(dimensions_list[index])*fx)/bb_width)
        str_distance = num2words(int(distance))
        return str_distance
    else:
        return None

  
def get_posistion(bb_width,x_origin):
     
    if (bb_width>=bb_threshold):
        direction = "straight ahead"
               
    else:
        tolerance = image_width * 0.1  # For 240 pixels, tolerance is 24 pixels   
        
        if abs(x_origin - center_of_frame) < tolerance:
            pos = "straight ahead"
        elif x_origin < center_of_frame:
            pos = "to your left"
        else:
            pos = "to your right"
            
        return pos
    
    
    
    
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue
    

    results = model.predict(frame, imgsz=240, conf = 0.5)
    object_detected = results[0].boxes.data #Raw data set as a numpy array
    data_set = pd.DataFrame(object_detected).astype("float") #use Pandas to convert array to a dataframe
    limit = data_set.head(max_det)#Limit number of detected objects by our model at a single time


    for index, row in limit.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        name_index = int(row[5])
        name = class_list[name_index]
        bb_width = x1-x2
        x_origin = abs(x1-x2/2)
        
        #Check if the object deteced by our model is one of the objects we plan on announcing
        if name not in objects_of_interest:
            continue
        
        #Check to see if the object has already been detetced in the last 10 seconds to avoid spamming 
        elif name in prev_object and (time.time()-prev_object[name]<cooldown_time):
            continue
        
    
        #Check to see if the objects name has been confirmed throughout multiple frames if so go ahead with the announcement
        elif confirm_object(name)==True:
            
            
            #Get the index of the object
            index_of_object = objects_of_interest.index(name)
            
            #Append the object to dictonary
            prev_object[name] = time.time()
            
            #Draw the bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{name}', (x1, y1), 1, 1)
            
            #Calculate the distance
            distance = get_distance(bb_width, x_origin, index_of_object)
            
            #Get the approximate posistion
            posistion = get_posistion(bb_width, x_origin)
             
            #Make the announcememnt
            if distance != None:
                engine.say(f"There is a {name}, {distance} centimeters {posistion}.") #Add the distance here if bug is fixed
                engine.runAndWait()
                
            else:
                engine.say(f"There is a {name}, {posistion}.")
                engine.runAndWait()
            
          
            
    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count/ elapsed_time

    # Display FPS on frame
    cvzone.putTextRect(frame, f'FPS: {round(fps, 2)}', (10, 30), 1,1)

    cv2.imshow("FRAME", frame)#Open a new window for camera called frame

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()

