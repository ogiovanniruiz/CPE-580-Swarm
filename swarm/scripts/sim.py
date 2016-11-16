#!/usr/bin/python2.7

import pygame
from pygame.locals import *
import numpy as np
from geometry_msgs.msg import Twist
import rospy
import sys
import random

red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
white = [255, 255, 255]
black = [0, 0, 0]

SCREENSIZE = [1500, 1000]  # Size of our output display
screenBGColor = black  # White background

N = 20  # Number of Robots
D = 2  # Dimension of search space
M = 0# Number of Obstacles
c = 0.0005  # iteration rate
c0 = 0.005  # leader iteration rate

A = np.ones((N,N), dtype=np.int)#np.random.rand(N,N) #Agacency Matrix is Randomized for now

collidables = []

bots = []

class Swarm_Simulation:
    def __init__(self):
        rospy.init_node('sim_node', anonymous=True)

        # Subscribe and Publish to Topics
        rospy.Subscriber("motor_control", Twist, self.array_callback)

        rospy.loginfo("Simulation Node is loaded...")

        pygame.init()  # Shall we begin?
        self.screen = pygame.display.set_mode(SCREENSIZE)
        self.screen.fill(screenBGColor)
        pygame.display.set_caption("Rendezvous Simulation")

        self.P = 1000 * np.random.rand(D, N)  # Initial positions of robots in a 800*800 unit space
        self.Pn = np.zeros((D, N), dtype=np.int)  # Position bucket for Greens

        self.P0 = 1000 * np.random.rand(D, 1)  #Initial position of RED Leader
        self.P0n = np.zeros((D, 1), dtype=np.int) # Position bucket for RED Leader

        self.vertical = 0 #Joy stick controls
        self.horizontal = 0

        self.target = [0,0]

        self.r = rospy.Rate(10)

        for i in range(N):

            bots.append(Circles(self.Pn[0,i], self.Pn[1,i], 3, 3, green))

        #collidables.extend(bots)
        Maze()
        self.Run()


    def array_callback(self, twist_array):

        self.vertical = -twist_array.linear.x
        self.horizontal = twist_array.angular.z

    def Run(self):
        while not rospy.is_shutdown():

            self.target = [[self.horizontal * 1000 + 750], [self.vertical * 1000 + 500]]  # final position of leader
            self.Pn = self.P
            self.P0n = self.P0
            self.red = pygame.Rect(self.P0[0,0],self.P0[1,0],2,2)


            self.P0n = self.P0n - c0 * (self.P0 - self.target)

            pygame.draw.circle(self.screen, black, [int(self.P0[0, 0]), int(self.P0[1, 0])], 7, 1)

            for i in range(N):
                for j in range(N):
                    for k in range(len(collidables)):

                        pygame.draw.circle(self.screen, black, [int(self.Pn[0, i]), int(self.Pn[1, i])], 7, 1)

                        if bots[i].rect.colliderect(collidables[k].rect):
                            #print(bots[3].rect)
                            print ("Collided")
                            print(collidables[k].rect)
                            self.Pn[:, i] = self.Pn[:, i] + c * (self.P[:, j] + self.P0[:, 0]) * A[i,j]
                            #self.Pn[:, i] = self.Pn[:, i] - c * 0.01 * (self.P[:, j] + self.P[:, i]) * A[i,j] + c * (self.P[:, j] + self.P0[:, 0]) * A[i,j]
                        else:
                            self.Pn[:, i] = self.Pn[:, i] + c * (self.P[:, j] + self.P0[:, 0]) * A[i, j]
                            #self.Pn[:, i] = self.Pn[:, i] + c * 0.01 * (self.P[:, j] - self.P[:, i]) * A[i, j] - c * (self.P[:, j] - self.P0[:, 0]) * A[i, j]

                        bots[i].rect.x = self.Pn[0,i]
                        bots[i].rect.y = self.Pn[1,i]
                        bots[i].x = self.Pn[0,i]
                        bots[i].y = self.Pn[1,i]

            for k in collidables:
                k.draw(self.screen)  # Draw all obstacles

            for k in bots:
                k.draw(self.screen) # Draw all Bots

            self.P = self.Pn
            self.P0 = self.P0n

            pygame.draw.circle(self.screen, red, [int(self.P0[0, 0]), int(self.P0[1, 0])], 7, 1)
            pygame.display.update()


class Collidable:
    def __init__(self, x, y, w, h, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, [self.x, self.y, self.w, self.h], 6)


def Maze():

        #Four Walls
        collidables.append(Collidable(0, 0, SCREENSIZE[0], 6, white))
        collidables.append(Collidable(0, 0, 6, SCREENSIZE[1], white))
        collidables.append(Collidable(SCREENSIZE[0] - 4, 0, 6, SCREENSIZE[1], white))
        collidables.append(Collidable(0, SCREENSIZE[1] - 4, SCREENSIZE[0], 6, white))

        for x in range(M):
                x = random.randint(0,SCREENSIZE[0])
                y = random.randint(0,SCREENSIZE[1])

                height = random.randint(1,3)
                width = random.randint(1,300)
                collidables.append(Collidable(x, y, height, width, white))

        for x in range(M):
                x = random.randint(0,SCREENSIZE[0])
                y = random.randint(0,SCREENSIZE[1])

                height = random.randint(1,300)
                width = random.randint(1,3)
                collidables.append(Collidable(x, y, height, width, white))

class Circles():
    def __init__(self, x, y, w, h, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.rect = pygame.Rect(x, y, w, h)
    def draw(self,screen):
        pygame.draw.circle(screen, self.color, [int(self.x), int(self.y)], 7, 1)



def main(args):
    try:
        Swarm_Simulation()

    except KeyboardInterrupt:
        print(" Shutting down simulation...")
        pygame.quit()

if __name__ == '__main__':
    main(sys.argv)

