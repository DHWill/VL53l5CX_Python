#!/usr/bin/env python3

import time
import vl53l5cx_ctypes as vl53l5cx

import numpy as np
from PIL import Image
from matplotlib import cm
import cv2
import socket
import json

ROTATE_TOF = -1 #1=90, 2=180 etc
MIN_DIST = 100
MAX_DIST = 1500
DEBUG = True
UDP_PORT = 6009


def cullReading_to_uInt8(minDist, maxDist, frameToCull):
    _frameToCull = np.where((frameToCull < minDist) | (frameToCull > maxDist), 0, frameToCull)
    range = maxDist - minDist
    _frameToCull -= minDist     #offsetToRange
    _frameToCull *= (255.0 /range)
    _frameToCull = np.clip(_frameToCull, 0, 255)    #clipoffset
    _frameToCull = np.where((_frameToCull > 0), (_frameToCull *-1 + 255), 0)    #invert values in range
    return _frameToCull.astype('uint8')

def uInt8_toBin(frameToBin):
    _frameToBin = np.where((frameToBin > 0), 255, 0)
    return _frameToBin.astype('uint8')


def calcOpticalFlow(frame1, frame2):

            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 1, 1, 1, 1, 1.2, 0)

            # Extract horizontal (u) and vertical (v) flow components
            flow_u = flow[..., 0]  # Horizontal component of flow
            flow_v = flow[..., 1]  # Vertical component of flow

            # Calculate the average of the flow vectors
            average_u = np.mean(flow_u)  # Average of the horizontal components
            average_v = np.mean(flow_v)  # Average of the vertical components

            # The average flow vector is (average_u, average_v)
            average_vector = np.array([average_u, average_v])
            # print("average_vector", average_vector)

            # Calculate the magnitude and angle of the average direction vector
            magnitude = np.sqrt(average_u**2 + average_v**2)  # Magnitude of the average vector
            angle = np.arctan2(average_v, average_u)  # Angle of the average vector in radians

            # Convert angle to degrees if needed
            angle_deg = np.degrees(angle)
            # print("angle_deg", angle_deg)

            # Visualize the optical flow
            # Create an HSV image to represent the flow
            hsv = np.zeros_like(cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR))
            hsv[...,1] = 255  # Set saturation to max

            # Convert flow to polar coordinates
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2  # Set hue according to flow direction
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Set value according to flow magnitude

            # Convert HSV to BGR for visualization
            flow_visual = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return flow_visual


def find_blob_centroid(grid):
    """Find the centroid of the blob using contours."""
    # Convert grid to a proper format for OpenCV (binary image)
    grid = np.array(grid, dtype=np.uint8) * 255  # Convert 0/1 grid to 0/255 binary image
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None  # No blob found
    
    # Assume the largest contour is the blob
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute the centroid using moments
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None  # Prevent division by zero
    
    cx = int(M["m10"] / M["m00"])  # x-coordinate of centroid
    cy = int(M["m01"] / M["m00"])  # y-coordinate of centroid
    
    return (cx, cy)

def calculate_movement_direction(previous_centroid, current_centroid):
    """Calculate the direction of movement based on two centroids."""
    if previous_centroid is None or current_centroid is None:
        return "No movement"  # No movement detected
    
    dx = current_centroid[0] - previous_centroid[0]
    dy = current_centroid[1] - previous_centroid[1]
    
    # Determine direction based on dx, dy
    if dx == 0 and dy == 0:
        return "No movement"
    elif dx == 0:
        return "Up" if dy < 0 else "Down"
    elif dy == 0:
        return "Left" if dx < 0 else "Right"
    else:
        if dx < 0 and dy < 0:
            return "Up-Left"
        elif dx > 0 and dy < 0:
            return "Up-Right"
        elif dx < 0 and dy > 0:
            return "Down-Left"
        elif dx > 0 and dy > 0:
            return "Down-Right"

def calcLeftRight(frame1, frame2):
    width, height = frame1.shape
    sumX1 = np.sum(frame1, axis=0)
    sumX2 = np.sum(frame2, axis=0)

    #first and last highest in flattened 
    max_index_1 = np.argmax(sumX1)
    max_index_last_1= len(sumX1) - 1 - np.argmax(sumX1[::-1])
    if(max_index_last_1 != max_index_1):
        width1 = max_index_last_1 - max_index_1
        centroid_1 = (width1 * 0.5) + max_index_1
    else:
        width1 = max_index_1
        centroid_1 = width1

    
    max_index_2= np.argmax(sumX2)
    max_index_last_2 = len(sumX2) - 1 - np.argmax(sumX2[::-1])
    
    if(max_index_last_2 != max_index_2):
        width2 = max_index_last_2 - max_index_2
        centroid_2 = (width2 * 0.5) + max_index_2
    else:
        width2 = max_index_2
        centroid_2 = width2

    direction = ""
    if(centroid_1 == centroid_2):
        direction = "Still"
    elif(centroid_1 > centroid_2):
        direction = "Right"
        pass
    else:
        direction = "Left"
        pass
    
    return ({'direction': direction, 'centroid': centroid_1 ,'distance': sumX1[max_index_1]})


def calc_centroid(frame1):
    sumX1 = np.sum(frame1, axis=0)

    #first and last highest in flattened (tallest and closest object)
    max_index = np.argmax(sumX1)
    max_index_last= len(sumX1) - 1 - np.argmax(sumX1[::-1])
    if(max_index_last != max_index):
        width1 = max_index_last - max_index
        centroid = (width1 * 0.5) + max_index
    else:
        width1 = max_index
        centroid = width1
    
    return (centroid)
    

def detect_blob_movement(previous_grid, current_grid):
    """Detect blob movement between two 8x8 grids."""
    previous_centroid = find_blob_centroid(previous_grid)
    current_centroid = find_blob_centroid(current_grid)
    
    direction = calculate_movement_direction(previous_centroid, current_centroid)
    return direction

def get_distance_from_blob_centroid(grid, centroid):
    #Sometimes the centre of the blob detected is blank, so average the square around it 
    distance = 0
    width, height = grid.shape
    if(centroid != None):
        centroid = np.array(centroid)
        distance = grid[(centroid[0], centroid[1])]
        if(distance != 0):
            return distance
        else:
            vecsToCheck = []
            vecsToCheck.append([0, 1])
            vecsToCheck.append([1, 1])
            vecsToCheck.append([1, 0])
            vecsToCheck.append([0, -1])
            vecsToCheck.append([-1, -1])
            vecsToCheck.append([-1, 0])
            
            nCheck = 0
            nAv = 0
            for vec in vecsToCheck:
                vecToCheck = (centroid + vec)
                if(vecToCheck[0] < width and vecToCheck[0] > 0):
                    if(vecToCheck[1] < height and vecToCheck[1] > 0):
                        val = grid[(vecToCheck[0], vecToCheck[1])]
                        if(val > 0):
                            nAv += val
                            nCheck += 1
            if(nCheck >0):
                distance = nAv /nCheck
    
    return distance
            




class vl53_TOF_Sensor():
    def setup(self, rotate=ROTATE_TOF):
        print("Uploading firmware, please wait...")
        self.vl53 = vl53l5cx.VL53L5CX()
        print("Done!")
        self.vl53.set_resolution(8 * 8)
        self.vl53.set_ranging_frequency_hz(30)
        self.vl53.set_integration_time_ms(20)
        self.lastFrame = []
        self.rotate = rotate
        self.isInit = True
        return self.vl53

    def isDataReady(self):
        return self.vl53.data_ready()
    
    def startSensor(self):
        self.vl53.start_ranging()

    def get_frame_uint8(self):
        data = self.vl53.get_data()
        if(self.isInit):
            frame = np.flipud(np.array(data.distance_mm).reshape((8, 8))).astype('float64')
            self.lastFrame = frame.copy()
            self.isInit = False
        else:
            frame = np.flipud(np.array(data.distance_mm).reshape((8, 8))).astype('float64')

        cleanFrame = np.where((frame == self.lastFrame), 0, frame)
        self.lastFrame = frame.copy()
        frame_uint8 = cullReading_to_uInt8(MIN_DIST, MAX_DIST, cleanFrame)
        frame_uint8 = np.rot90(frame_uint8, k=self.rotate)
        time.sleep(0.01)  # Avoid polling i2c data is ready *too* fast
        return frame_uint8



def main():
    
    target_ip = '127.0.0.1'
    target_port = UDP_PORT
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    tofSensor = vl53_TOF_Sensor()
    tofSensor.setup()
    tofSensor.startSensor()
    while(tofSensor.isDataReady()==0):
        time.sleep(0.01)
    tof_frame = tofSensor.get_frame_uint8()
    last_tof_frame = tof_frame.copy()
    vSpeed = 0.5
    velocity = 0
    posX = 0
    


    messageInfo = {'inShot': "False", 'centroid.x': "0" , 'centroid.y': "0" ,'distance': "0.0", 'xAverage': "0.5" }
    while True:
        if(tofSensor.isDataReady()):
            tof_frame = tofSensor.get_frame_uint8()
            width, height = tof_frame.shape
            
            
            #binaryChop
            binFrame1 = uInt8_toBin(tof_frame)
            binFrame2 = uInt8_toBin(last_tof_frame)

            #Blob Detection
            blobCentre = find_blob_centroid(tof_frame)

            
            destX = calc_centroid(binFrame1) / width
            destX *=2.
            destX -= 1.
            velocity = destX - posX
            posX += (velocity * vSpeed)
            posXI = posX +1
            posXI /=2
            
            velocity_to_send = 0
            if(abs(velocity) > 0.25):
                velocity_to_send = velocity
            
            #Optical Flow
            # flow_visual = calcOpticalFlow(tof_frame, last_tof_frame)

            if(blobCentre!=None):
                messageInfo["inShot"] = int(1)
                messageInfo["centroid.x"] = blobCentre[0]
                messageInfo["centroid.y"] = blobCentre[1]
                messageInfo["distance"] = int(get_distance_from_blob_centroid(tof_frame, blobCentre))
                messageInfo["xAverage"] = float(posXI)
            else:
                messageInfo['inShot'] = int(0)
            
            flow_visual = tof_frame

            
            message = json.dumps(messageInfo).encode('utf-8')
            sock.sendto(message, (target_ip, target_port))

            # Resize images for visualization
            if(DEBUG):
                for mess in messageInfo.items():
                    print(mess)
                
                flow_resized = cv2.resize(flow_visual, (256, 256), interpolation=cv2.INTER_NEAREST)
                frame1_resized = cv2.resize(tof_frame, (256, 256), interpolation=cv2.INTER_NEAREST)
                binFrame = cv2.resize(binFrame1, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Frame ', frame1_resized)
                cv2.imshow('BinaryFrame ', binFrame)
                cv2.imshow('Optical Flow', flow_resized)
            # cv2.imshow('binary_cull', binary_cull_resized)
            last_tof_frame = tof_frame.copy()
            k = cv2.waitKey(10) & 0xff
            # Close all windows after keypress
            if k == 27: #EXIT
                break
    
    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    main()
