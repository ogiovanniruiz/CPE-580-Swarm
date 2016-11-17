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

N = 50  # Number of Robots
D = 2  # Dimension of search space
M = 20# Number of Obstacles
c = 0.005  # iteration rate
dt = 5  # leader iteration rate

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

        self.vel_y = 0 #Joy stick controls
        self.vel_x = 0

        self.rand_speeds = np.random.rand(D,N)

        for i in range(N):

            bots.append(Circles(self.Pn[0,i], self.Pn[1,i], 30, 30, green))

        self.Maze()

        self.Run()


    def array_callback(self, twist_array):

        self.vel_y = -twist_array.linear.x
        self.vel_x = twist_array.angular.z

        #print (self.vel_x)

    def Run(self):
        while not rospy.is_shutdown():

            self.P0n = self.P0

            pygame.draw.circle(self.screen, black, [int(self.P0n[0, 0]), int(self.P0n[1, 0])], 7, 1)

            self.P0n[:,0] = self.P0n[:,0] + [(self.vel_x *dt),(self.vel_y * dt)]

            self.red_leader = pygame.Rect(int(self.P0n[0, 0]), int(self.P0n[1, 0]), 7, 7)
            self.los = pygame.Rect(int(self.P0n[0, 0]), int(self.P0n[1, 0]), 200, 200)

            self.Pn = self.P

            for i in range(N):
                for k in range(len(collidables)):

                    if self.red_leader.colliderect(collidables[k].rect):

                        self.P0n[:, 0] = self.P0n[:, 0] + [(-self.vel_x * dt), (-self.vel_y * dt)]
                        self.red_leader = pygame.Rect(int(self.P0[0, 0]), int(self.P0[1, 0]), 10, 10)
                        self.los = pygame.Rect(int(self.P0n[0, 0]), int(self.P0n[1, 0]), 35, 35)

                    else:

                        self.P0n = self.P0n
                        self.red_leader = pygame.Rect(int(self.P0n[0, 0]), int(self.P0n[1, 0]), 10, 10)
                        self.los = pygame.Rect(int(self.P0n[0, 0]), int(self.P0n[1, 0]), 35, 35)


                    pygame.draw.circle(self.screen, black, [int(self.Pn[0, i]), int(self.Pn[1, i])], 7, 1)

                    self.Pn[:, i] = self.Pn[:, i] + [self.rand_speeds[:, i] *0.1]

                    bots[i].rect.x = self.Pn[0, i]
                    bots[i].rect.y = self.Pn[1, i]

                    if bots[i].rect.colliderect(self.los):
                        self.Pn[:, i] = self.Pn[:, i] - c * (self.P[:, i] - self.P0[:, 0])
                        self.rand_speeds[:,i] = -0.1 * self.rand_speeds[:,i]
                        num_cap = len(self.los.collidelistall(bots))
                        print (num_cap)
                    else:
                        self.Pn[:, i] = self.Pn[:, i] + [self.rand_speeds[:, i] * 0.1]


                    if bots[i].rect.colliderect(collidables[k].rect):
                        self.rand_speeds[:, i] = - self.rand_speeds[:,i]
                        bots[i].rect.x = self.P[0, i]
                        bots[i].rect.y = self.P[1, i]

                    else:
                        self.Pn[:, i] = self.Pn[:, i]
                        bots[i].rect.x = self.P[0, i]
                        bots[i].rect.y = self.P[1, i]


                    bots[i].x = self.P[0,i]
                    bots[i].y = self.P[1,i]

            for k in collidables:
                k.draw(self.screen)  # Draw all obstacles

            for k in bots:
                k.draw(self.screen) # Draw all Bots

            self.P = self.Pn
            self.P0 = self.P0n

            pygame.draw.circle(self.screen, red, [int(self.P0[0, 0]), int(self.P0[1, 0])], 7, 1)
            pygame.display.update()

    def Maze(self):

        # Four Walls
        collidables.append(Collidable(0, 0, SCREENSIZE[0], 10, white))
        collidables.append(Collidable(0, 0, 10, SCREENSIZE[1], white))
        collidables.append(Collidable(SCREENSIZE[0] - 4, 0, 10, SCREENSIZE[1], white))
        collidables.append(Collidable(0, SCREENSIZE[1] - 4, SCREENSIZE[0], 10, white))

        for x in range(M):
            x = random.randint(0, SCREENSIZE[0])
            y = random.randint(0, SCREENSIZE[1])

            height = random.randint(9, 10)
            width = random.randint(1, 500)
            collidables.append(Collidable(x, y, height, width, white))

        for x in range(M):
            x = random.randint(0, SCREENSIZE[0])
            y = random.randint(0, SCREENSIZE[1])

            height = random.randint(1, 500)
            width = random.randint(9, 10)
            collidables.append(Collidable(x, y, height, width, white))




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

