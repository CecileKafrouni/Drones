#!/usr/bin/env python
#from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from collections import deque
from imutils.video import VideoStream
import argparse
import imutils

from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

import time

# variables
position_x_target = 310
position_y_target = 180

taille_target = 10




class MoveDrone():



	def __init__(self):

		self.takeoff_pub = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10) # TODO put the takeoff topic name here
		self.landing_pub = rospy.Publisher("/ardrone/land", Empty, queue_size=10) # TODO put the landing topic name here

		self.move_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10) # Publish commands to drone 

		im = image_converter()
	def move_drone(self, speed=[0.0, 0.0, 0.0], orient=[0.0, 0.0, 0.0]):
		
		
		# TODO: fill the velocity fields here with the desired values
		vel_msg = Twist()
		vel_msg.linear.x = speed[0]
		vel_msg.linear.y = speed[1]
		vel_msg.linear.z = speed[2]
	

		# TODO: fill the angulare velocities here with the desired values

		#vel_msg.angular.x = speed[0]
		#vel_msg.angular.y = speed[1]
		#vel_msg.angular.z = speed[2]

		self.move_pub.publish(vel_msg)

		return 0
        '''
	def move_up_down(self):
		#movedrone.move_drone(speed = [0.0,0.0,0.1])
	   	#movedrone.move_drone(speed = [0.1,0.0,0.0])
	   	#movedrone.move_drone(speed = [0.0,0.1,0.0])

	#def move_right_left(self):

	#def move_straight_back(self):

	def rotate_right_left(self):
        '''
	def takeoff_drone(self):
		empty_msg = Empty()

        	self.takeoff_pub.publish(empty_msg)
		print("takeoff")

	def land_drone(self):
		empty_msg = Empty()
		# TODO: send landing command to the drone
        	self.landing_pub.publish(empty_msg)

class image_converter:

  flag = False
  x = 0
  y = 0

  def __init__(self):

    # pour le moment on en a pas besoin
    #self.image_pub = rospy.Publisher("image_topic_2",Image) #pour publier les messages avec 
    #le nom du topic ou on veut publier des messages et le 2eme argument cest le type de message
    #self.movedrone =movedrone 
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/ardrone/front/image_raw",Image,self.callback) # pour ecouter les messages, avec le nom du canal quon veut ecouter

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") #bgr pour dire que cets une image rgb
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (position_x_target,position_y_target), taille_target, 255) #une fois quon a l'image on cree un cercle dessus
      hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

# pour detecter la couleur verte 
      low = np.array([29, 86, 6])
      high = np.array([64, 240, 240])


      ap = argparse.ArgumentParser()
      #ap.add_argument("-v", "--video", help="path to the (optional) video file")
      ap.add_argument("-b", "--buffer", type=int, default=16, help="max buffer size")
      args = vars(ap.parse_args())
      
      pts = deque(maxlen=args["buffer"])

      blurred = cv2.GaussianBlur(cv_image, (11, 11), 0)
      hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

      

      image_mask = cv2.inRange(hsv, low, high)

      image_mask = cv2.erode(image_mask, None, iterations=2)
      image_mask = cv2.dilate(image_mask, None, iterations=2)

      output = cv2.bitwise_and(cv_image, cv_image, mask=image_mask)

      cnts = cv2.findContours(image_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      center = None

      if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
         c = max(cnts, key=cv2.contourArea)
         ((x, y), radius) = cv2.minEnclosingCircle(c)
         M = cv2.moments(c)
         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

# only proceed if the radius meets a minimum size
         if radius > 10:
             self.flag = True
             #movedrone.takeoff
	# draw the circle and centroid on the frame,
	# then update the list of tracked points
             cv2.circle(cv_image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
             cv2.circle(cv_image, center, 5, (0, 0, 255), -1)
	     self.x = int(x)
	     self.y = int(y)
             #print('position : {}, {}'.format(int(x), int(y)))
      pts.appendleft(center)



      for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
          if pts[i - 1] is None or pts[i] is None:
               continue
# otherwise, compute the thickness of the line and
		# draw the connecting lines
          thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
          cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

# show the frame to our screen
      cv2.imshow("Frame", cv_image)
      key = cv2.waitKey(1) & 0xFF

      #cv2.imshow("Image mask", image_mask)
      #cv2.imshow("Color tracking mask", output)
      

    #cv2.imshow("Image window", cv_image) #afficher l'image
    cv2.waitKey(3)



  def returnFlag(self):
    #print('flag', self.flag)
    return self.flag

'''
# fait partie du publisher
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
'''
def main(args):
  
  ic = image_converter()
  #rospy.init_node('image_converter', anonymous=True)
  movedrone = MoveDrone()
  
  #rospy.init_node('image_converter', anonymous=True)
  rospy.init_node('MoveDrone', anonymous=True)

 
  # on le fait decoller
  t1 = time.time()

  while((time.time()-t1)< 5):
     movedrone.takeoff_drone()
     print("takeoff")

  t2 = time.time()

  while((time.time()-t2)< 30): 
    # si il trouve du vert
     if ic.flag:
        #print('trouver vert')
        print('position : {}, {}'.format(ic.x, ic.y))

      # si la couleur verte est sur la droite du rond bleu  
     	if(ic.x < 310 and ic.y < 180):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.x-position_x_target)/(0.1*1000))): # pour avoir combien de temps il doit parcourir v = d/t donc t=d/v
              movedrone.move_drone(speed = [0.0,0.1,0.0]) # bouge vers la droite
	      print("bouge vers droite")
	      movedrone.move_drone(speed = [0.0,0.0,0.1]) # bouge vers le haut
	      print("bouge vers haut")
            ic.flag = False
	    print(ic.flag)

	elif(ic.x < 310 and ic.y > 180):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.y-position_y_target)/(0.1*1000))):
	      movedrone.move_drone(speed = [0.0,0.1,0.0]) # bouge vers la droite
	      print("bouge vers droite")
              movedrone.move_drone(speed = [0.0,0.0,-0.1]) # bouge vers le bas
	      print("bouge vers bas")
            ic.flag = False
	    print(ic.flag)


      # sur la gauche du rond bleu  
        elif(ic.x > 310 and ic.y < 180):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.x-position_x_target)/(0.1*1000))):
               movedrone.move_drone(speed = [0.0,-0.1,0.0]) # bouge vers la gauche
	       print("bouge vers gauche")
	       movedrone.move_drone(speed = [0.0,0.0,0.1]) # bouge vers le haut
	       print("bouge vers haut")
            ic.flag = False
	    print(ic.flag)

	elif(ic.x > 310 and ic.y > 180):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.x-position_x_target)/(0.1*1000))):
               movedrone.move_drone(speed = [0.0,-0.1,0.0]) # bouge vers la gauche
	       print("bouge vers gauche")
	       movedrone.move_drone(speed = [0.0,0.0,-0.1]) # bouge vers le bas
	       print("bouge vers bas")
            ic.flag = False
	    print(ic.flag)

	elif(ic.x < 310):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.x-position_x_target)/(0.1*1000))): # pour avoir combien de temps il doit parcourir v = d/t donc t=d/v
               movedrone.move_drone(speed = [0.0,0.1,0.0]) # bouge vers la droite
            ic.flag = False
	    print(ic.flag)

      # sur la gauche du rond bleu  
        elif(ic.x > 310):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.x-position_x_target)/(0.1*1000))):
              movedrone.move_drone(speed = [0.0,-0.1,0.0]) # bouge vers la gauche
            ic.flag = False
	    print(ic.flag)

      # au dessus du rond bleu
        elif(ic.y < 180):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.y-position_y_target)/(0.1*1000))):
              movedrone.move_drone(speed = [0.0,0.0,0.1]) # bouge vers le haut
            ic.flag = False
	    print(ic.flag)

      # en dessous du rond bleu
        elif(ic.y > 180):
            t3 = time.time()
            while((time.time()-t3) < (abs(ic.y-position_y_target)/(0.1*1000))):
              movedrone.move_drone(speed = [0.0,0.0,-0.1]) # bouge vers le bas
            ic.flag = False
	    print(ic.flag)


  # puis on le fait atterir
  movedrone.land_drone()
  print("land")

  	
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)