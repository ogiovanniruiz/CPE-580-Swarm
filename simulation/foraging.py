import pygame
from pygame.locals import *

import numpy as np

red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
white = [255, 255, 255]
black = [0, 0, 0]

N = 40  # Number of Robots
D = 2  # Dimension of search space
dt = 0.005  # iteration rate
running = True

SCREENSIZE = [1000, 1000]  # Size of our output display
NESTSIZE = [50,50]
RESOURCE_SIZE = [25,25]
RESOURCE_LOCATION = np.random.rand(D,1) * SCREENSIZE[0]

nest = pygame.Rect(SCREENSIZE[0]/2 - NESTSIZE[0]/2,SCREENSIZE[1]/2 - NESTSIZE[1]/2,NESTSIZE[0],NESTSIZE[1])
food = pygame.Rect(RESOURCE_LOCATION[0], RESOURCE_LOCATION[1],RESOURCE_SIZE[0],RESOURCE_SIZE[1])

los = []

los_size = 200

screenBGColor = black  # Background Color

A = np.random.rand(N,N)

#print(A)

class Swarm_Simulation:
    def __init__(self):

        pygame.init()  # Shall we begin?
        self.screen = pygame.display.set_mode(SCREENSIZE)
        self.screen.fill(screenBGColor)
        pygame.display.set_caption("Rendezvous Simulation")

        self.P = (SCREENSIZE[0]/2)* np.ones((D, N),dtype=np.int)  # Initial positions of robots in screen
        self.Pn = np.zeros((D, N), dtype=np.int)  # Position bucket I THINK...

        for i in range(N):
            los.append(Detectable(self.Pn[0,i]-los_size/2, self.Pn[1,i]-los_size/2, los_size, los_size, green))

        self.Run()


    def Run(self):

        while (running):

            pygame.draw.rect(self.screen, white,[SCREENSIZE[0]/2 - NESTSIZE[0]/2,SCREENSIZE[1]/2 - NESTSIZE[1]/2,NESTSIZE[0],NESTSIZE[1]],5)

            pygame.draw.rect(self.screen, red,[RESOURCE_LOCATION[0], RESOURCE_LOCATION[1],RESOURCE_SIZE[0],RESOURCE_SIZE[1]],5)

            self.Pn = self.P

            for i in range(N):
                #for j in range(N):

                    

                    pygame.draw.circle(self.screen, black, [int(self.P[0, i]), int(self.P[1, i])], 7, 1)

                    
                    if los[i].rect.colliderect(food):
                        rand_speeds[:,i] = [0,0]

                        pygame.draw.circle(self.screen, red, [int(self.P[0, i]), int(self.P[1, i])], 7, 1)
                        print("Found!!!!")

                    else:
                        rand_speeds = np.random.randint(1000, size = (D,N)) - 400

                        self.Pn[:, i] = self.Pn[:, i] + [rand_speeds[:,i]*dt]

                        pygame.draw.circle(self.screen, green, [int(self.P[0, i]), int(self.P[1, i])], 7, 1)


                    #self.Pn[:, i] = self.Pn[:, i] + c* (self.P[:, j] - self.P[:, i]) * A[i,j]



                    los[i].rect.x = self.Pn[0, i]-los_size/2
                    los[i].rect.y = self.Pn[1, i]-los_size/2

            self.P = self.Pn

            pygame.display.update()


class Detectable:
    def __init__(self, x, y, w, h, color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, [self.x, self.y, self.w, self.h], 6)


if __name__ == '__main__':

    try:
        Swarm_Simulation()

    except KeyboardInterrupt:
        print (" Shutting down simulation...")
        pygame.quit()
