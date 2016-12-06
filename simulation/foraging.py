import pygame
from pygame.locals import *

import numpy as np

import time

red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
white = [255, 255, 255]
black = [0, 0, 0]
yellow = [255, 255, 0]
orange = [255, 165, 0]
gold = [255, 215, 0]

N = 50  # Number of Robots
D = 2  # Dimension of search space
dt = 0.005  # iteration rate
running = True

SCREENSIZE = [1000, 1000]  # Size of our output display
NESTSIZE = [50,50]
RESOURCE_SIZE = [50,50]
RESOURCE_LOCATION = np.random.rand(D,1) * SCREENSIZE[0]

#RESOURCE_LOCATION = [RESOURCE_SIZE[0]*4,SCREENSIZE[1]/2 - NESTSIZE[1]/2]
NEST_LOCATION = [SCREENSIZE[0]/2 - NESTSIZE[0]/2,SCREENSIZE[1]/2 - NESTSIZE[1]/2]

nest = pygame.Rect(NEST_LOCATION[0],NEST_LOCATION[1],NESTSIZE[0],NESTSIZE[1])
food = pygame.Rect(RESOURCE_LOCATION[0], RESOURCE_LOCATION[1],RESOURCE_SIZE[0],RESOURCE_SIZE[1])

los = []

los_size = 100

screenBGColor = black  # Background Color

bots = []

lines = []

target = []

A = np.random.rand(N,N)

class Swarm_Simulation:
    def __init__(self):

        pygame.init()  # Shall we begin?
        self.screen = pygame.display.set_mode(SCREENSIZE)
        self.screen.fill(screenBGColor)
        pygame.display.set_caption("Foraging Simulation")

        self.myfont = pygame.font.SysFont("monospace", 25)

        self.P = (SCREENSIZE[0]/2)* np.ones((D, N),dtype=np.int)  # Initial positions of robots in screen
        self.Pn = np.zeros((D, N), dtype=np.int)  # Position bucket I THINK...

        self.connected = False

        for i in range(N):

            bots.append(Circles(self.Pn[0,i], self.Pn[1,i], 30, 30, green))

        for i in range(N):

            los.append(Detectable(self.Pn[0,i]-los_size/2, self.Pn[1,i]-los_size/2, los_size, los_size, green,'walker',False,0))
        self.num_food_at_resource = 500

        self.num_food_at_nest = 0

        self.Run()

    def Run(self):

        while (running):

            label_nest_previous = self.myfont.render(str(self.num_food_at_nest), 1, black)

            self.screen.blit(label_nest_previous, (NEST_LOCATION[0]+NESTSIZE[0]/4, NEST_LOCATION[1]+NESTSIZE[1]/2))


            label_resource_previous = self.myfont.render(str(self.num_food_at_resource), 1, black)

            self.screen.blit(label_resource_previous, (RESOURCE_LOCATION[0]+ RESOURCE_SIZE[0]/4, RESOURCE_LOCATION[1]+RESOURCE_SIZE[1]/2))


            pygame.draw.rect(self.screen, white,[NEST_LOCATION[0],NEST_LOCATION[1],NESTSIZE[0],NESTSIZE[1]],5)

            pygame.draw.rect(self.screen, red,[RESOURCE_LOCATION[0], RESOURCE_LOCATION[1],RESOURCE_SIZE[0],RESOURCE_SIZE[1]],5)

            self.Pn = self.P

            for i in range(N):
                for j in range(N):

                    pygame.draw.circle(self.screen, black, [int(self.P[0, i]), int(self.P[1, i])], 7, 1)

                    if (los[i].mode == 'walker'):

                        rand_speeds = np.random.randint(1000, size = (D,N))- 400

                        self.Pn[:, i] = self.Pn[:, i] + [rand_speeds[:,i]*dt]

                        bots[i].color = green

                    elif (los[i].mode == 'gatherer'):

                        bots[i].color = blue

                    elif (los[i].mode == 'beacon') and (los[i].grad ==1):

                        bots[i].color = red

                        self.Pn[:,i] = self.Pn[:,i]

                        pygame.draw.aaline(self.screen, red, self.P[:,i],RESOURCE_LOCATION,1)

                    elif (los[i].mode == 'beacon') and (los[i].grad ==2):

                        bots[i].color = orange

                        self.Pn[:,i] = self.Pn[:,i]

                    elif (los[i].mode == 'beacon') and (los[i].grad ==3):

                        bots[i].color = gold

                        self.Pn[:,i] = self.Pn[:,i]

                    elif (los[i].mode == 'beacon') and (los[i].grad ==4):

                        bots[i].color = yellow

                        self.Pn[:,i] = self.Pn[:,i]
                    
                    elif (los[i].mode == 'beacon') and (los[i].grad ==5):

                        bots[i].color = white

                        self.Pn[:,i] = self.Pn[:,i]

                        pygame.draw.aaline(self.screen, bots[i].color, self.P[:,i],NEST_LOCATION,1)

                        self.connected = True
                    else:
                        print("ERROR_A")



                    if not self.connected:

                        if (los[i].rect.colliderect(food)):

                            los[i].mode = 'beacon'

                            los[i].grad = 1

                        elif (A[i,j] == 1) and (los[i].mode == 'walker') and (los[j].mode == "beacon"):

                            los[i].grad = los[j].grad + 1 

                            if los[i].grad > 5:
                                los[i].grad = 5

                            lines.append(Lines(self.P[:,i], self.P[:,j], bots[j].color))
                            
                            los[i].mode = 'beacon' 

                        elif (los[i].rect.colliderect(nest)) and (los[i].mode == "beacon"):

                            lines.append(Lines(NEST_LOCATION, self.P[:,i], bots[i].color)) 

                            self.connected = True            

                    elif self.connected:

                        if (A[i,j] == 1) and (los[i].mode == 'walker') and (los[j].mode == "beacon"):

                            los[i].mode = "gatherer"

                        elif (los[i].rect.colliderect(nest) and (los[i].mode == 'walker')):
                            
                            los[i].mode = "gatherer"


                        if  (los[i].mode == 'gatherer'):


                            if bots[i].rect.colliderect(food) and los[i].has_food == False:

                                los[i].has_food = True

                                self.num_food_at_resource -= 1

                                rand_speeds = np.random.randint(1000, size=(D, N)) - 500

                                self.Pn[1, i] = self.Pn[1, i] +  ((NEST_LOCATION[1] + NESTSIZE[1]*2 - self.P[1,i]))*dt + [rand_speeds[1,i]*dt]

                                self.Pn[0, i] = self.Pn[0, i] +  (NEST_LOCATION[0] + NESTSIZE[0]*2 - self.P[0,i])*dt + [rand_speeds[0,i]*dt]

                            elif bots[i].rect.colliderect(nest) and los[i].has_food == True :

                                los[i].has_food = False

                                self.num_food_at_nest += 1

                                rand_speeds = np.random.randint(1000, size=(D, N)) - 500

                                self.Pn[1, i] = self.Pn[1, i] +  (RESOURCE_LOCATION[1] + RESOURCE_SIZE[1]*3 - self.P[1,i])*dt + [rand_speeds[1,i]*dt]

                                self.Pn[0, i] = self.Pn[0, i] +  (RESOURCE_LOCATION[0] + RESOURCE_SIZE[0]*3 - self.P[0,i])*dt + [rand_speeds[0,i]*dt]


                            elif los[i].has_food == False:


                                rand_speeds = np.random.randint(1000, size=(D, N)) - 500

                                self.Pn[1, i] = self.Pn[1, i] + (RESOURCE_LOCATION[1] + RESOURCE_SIZE[1]*3- self.P[1, i]) * dt + [rand_speeds[1, i] * dt]

                                self.Pn[0, i] = self.Pn[0, i] + (RESOURCE_LOCATION[0]  + RESOURCE_SIZE[0]*3 - self.P[0, i]) * dt + [rand_speeds[0, i] * dt]

                            elif los[i].has_food == True:

                                rand_speeds = np.random.randint(1000, size=(D, N)) - 500

                                self.Pn[1, i] = self.Pn[1, i] +  ((NEST_LOCATION[1] + NESTSIZE[1]*2- self.P[1,i]))*dt + [rand_speeds[1,i]*dt]

                                self.Pn[0, i] = self.Pn[0, i] +  (NEST_LOCATION[0] + NESTSIZE[0]*2 - self.P[0,i])*dt + [rand_speeds[0,i]*dt]

                            else:

                                rand_speeds = np.random.randint(500, size=(D, N)) - 250

                                self.Pn[1, i] = self.Pn[1, i] + (RESOURCE_LOCATION[1] + RESOURCE_SIZE[1]*3 - self.P[1, i]) * dt + [rand_speeds[1, i] * dt]

                                self.Pn[0, i] = self.Pn[0, i] + (RESOURCE_LOCATION[0] + RESOURCE_SIZE[0]*3 - self.P[0, i]) * dt + [rand_speeds[0, i] * dt]


                    else:
                        print ("CONNECTION LOST")


                    

                    if (los[i].rect.colliderect(los[j].rect)):
                            
                        A[i,j] = 1
                    else:
                        A[i,j] = 0
                    
                    bots[i].x = self.P[0,i]
                    bots[i].y = self.P[1,i]

                    bots[i].rect.x = self.P[0,i]
                    bots[i].rect.y = self.P[1,i]

                    los[i].rect.x = self.P[0, i]-los_size/2
                    los[i].rect.y = self.P[1, i]-los_size/2


            for k in bots:
                k.draw(self.screen) # Draw all Bots
            
            for k in lines:
                k.draw(self.screen)

            self.P = self.Pn

            label_nest = self.myfont.render(str(self.num_food_at_nest), 1, white)

            self.screen.blit(label_nest, (NEST_LOCATION[0]+ NESTSIZE[0]/4, NEST_LOCATION[1]+NESTSIZE[1]/2))


            label_resource = self.myfont.render(str(self.num_food_at_resource), 1, red)

            self.screen.blit(label_resource, (RESOURCE_LOCATION[0]+ RESOURCE_SIZE[0]/4, RESOURCE_LOCATION[1]+RESOURCE_SIZE[1]/2))


            pygame.display.update()


class Detectable:
    def __init__(self, x, y, w, h, color,mode,has_food, grad):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.mode = mode
        self.has_food = has_food
        self.grad = grad
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

class Lines():
    def __init__(self, pos1, pos2, color):
        self.pos1 = pos1
        self.pos2 = pos2
        self.color = color

    def draw(self,screen):
        pygame.draw.aaline(screen, self.color, self.pos1, self.pos2,1)

if __name__ == '__main__':

    try:
        Swarm_Simulation()

    except KeyboardInterrupt:
        print (" Shutting down simulation...")
        pygame.quit()
