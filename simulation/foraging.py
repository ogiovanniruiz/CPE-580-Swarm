import pygame
from pygame.locals import *

import numpy as np

red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
white = [255, 255, 255]
black = [0, 0, 0]
yellow = [255, 255, 0]

N = 50  # Number of Robots
D = 2  # Dimension of search space
dt = 0.005  # iteration rate
running = True
c = 0.000000000000000000000000000000000005

SCREENSIZE = [1000, 1000]  # Size of our output display
NESTSIZE = [50,50]
RESOURCE_SIZE = [25,25]
RESOURCE_LOCATION = np.random.rand(D,1) * SCREENSIZE[0]

NEST_LOCATION = [SCREENSIZE[0]/2 - NESTSIZE[0]/2,SCREENSIZE[1]/2 - NESTSIZE[1]/2]

nest = pygame.Rect(NEST_LOCATION[0],NEST_LOCATION[1],NESTSIZE[0],NESTSIZE[1])
food = pygame.Rect(RESOURCE_LOCATION[0], RESOURCE_LOCATION[1],RESOURCE_SIZE[0],RESOURCE_SIZE[1])

los = []

los_size = 200

screenBGColor = black  # Background Color

bots = []

A = np.random.rand(D,N)

class Swarm_Simulation:
    def __init__(self):

        pygame.init()  # Shall we begin?
        self.screen = pygame.display.set_mode(SCREENSIZE)
        self.screen.fill(screenBGColor)
        pygame.display.set_caption("Rendezvous Simulation")

        self.P = (SCREENSIZE[0]/2)* np.ones((D, N),dtype=np.int)  # Initial positions of robots in screen
        self.Pn = np.zeros((D, N), dtype=np.int)  # Position bucket I THINK...

        self.connected = False

        for i in range(N):

            bots.append(Circles(self.Pn[0,i], self.Pn[1,i], 30, 30, green))

        for i in range(N):
            los.append(Detectable(self.Pn[0,i]-los_size/2, self.Pn[1,i]-los_size/2, los_size, los_size, green,0))

        self.Run()

    def Run(self):

        while (running):

            pygame.draw.rect(self.screen, white,[SCREENSIZE[0]/2 - NESTSIZE[0]/2,SCREENSIZE[1]/2 - NESTSIZE[1]/2,NESTSIZE[0],NESTSIZE[1]],5)

            pygame.draw.rect(self.screen, red,[RESOURCE_LOCATION[0], RESOURCE_LOCATION[1],RESOURCE_SIZE[0],RESOURCE_SIZE[1]],5)

            self.Pn = self.P

            for i in range(N):
                for j in range(N):

                    pygame.draw.circle(self.screen, black, [int(self.P[0, i]), int(self.P[1, i])], 7, 1)
                    
                    if (los[i].rect.colliderect(food)):

                        los[i].mode = 1

                        pygame.draw.line(self.screen, red, self.P[:,i],RESOURCE_LOCATION,1)

                    elif (los[i].rect.colliderect(los[j].rect)) and (los[j].mode == 1) and (los[i].mode == 0):
                            
                        los[i].mode = 1

                        los[j].mode = 2

                        pygame.draw.line(self.screen, red, self.P[:,i],self.P[:,j],1)

                    elif (los[i].rect.colliderect(nest)) and (los[i].mode == 2):
                        los[i].mode = 3

                        pygame.draw.line(self.screen, white, self.P[:,i],NEST_LOCATION,1)

                        self.connected == True
                               
                    elif los[i].mode == 0:
                        rand_speeds = np.random.randint(1000, size = (D,N))- 400

                        self.Pn[:, i] = self.Pn[:, i] + [rand_speeds[:,i]*dt]

                    else:

                        print ("CONNECTED")
                        #for i in range(N):
                            #los[i].mode = 3
                        target_speeds = [(self.P[0, i]**2 - RESOURCE_LOCATION[0,0]**2)**(0.5)*dt,(self.P[0, i]**2 - RESOURCE_LOCATION[0,0]**2)**(0.5)*dt ]

                        self.Pn[:, i] = self.Pn[:, i] - [target_speeds]

                        #self.Pn[0,i] = self.Pn[0,i] - dt * c * (self.P[0, i] - RESOURCE_LOCATION[0,0]) #* A[:,i]
                        #self.Pn[1,i] = self.Pn[1,i] - dt * c * (self.P[1, i] - RESOURCE_LOCATION[1,0]) #* A[:,i]

                    
                    if los[i].mode == 1:
                        bots[i].color = red

                    elif los[i].mode == 2:
                        bots[i].color = yellow

                    elif los[i].mode == 3:
                        bots[i].color = white

                    bots[i].x = self.P[0,i]
                    bots[i].y = self.P[1,i]

                    #self.Pn[:, j] = self.Pn[:, j] + c* (self.P[:, j] - self.P[:, i]) * A[i,j]

                    los[i].rect.x = self.Pn[0, i]-los_size/2
                    los[i].rect.y = self.Pn[1, i]-los_size/2


            for k in bots:
                k.draw(self.screen) # Draw all Bots

            self.P = self.Pn

            pygame.display.update()


class Detectable:
    def __init__(self, x, y, w, h, color,mode):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.mode = mode
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


if __name__ == '__main__':

    try:
        Swarm_Simulation()

    except KeyboardInterrupt:
        print (" Shutting down simulation...")
        pygame.quit()
