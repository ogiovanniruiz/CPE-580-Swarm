import pygame
from pygame.locals import *
import random

import numpy as np
from math import sin, cos, pi, floor, sqrt
from abc import abstractmethod

import MultiNEAT as NEAT


class Controller:
    @abstractmethod
    def __call__(self, senses : np.array) -> np.array:
        """Maps a set of inputs (numpy array) to a set of outputs (numpy array)"""
        pass

    @abstractmethod
    def _feedback(self, *args):
        pass

class DumbController:
    def __call__(self, senses : np.array) -> np.array:
        #return np.array([random.choice((-1, 1)) for i in range(2)])
        return [0, 0]

#CLASS MULTINEATCONTROLLER:j
params = NEAT.Parameters()
params.PopulationSize = 20
params.DynamicCompatibility = True
params.WeightDiffCoeff = 4.0
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 10
params.RouletteWheelSelection = False
params.RecurrentProb = 0.1
params.OverallMutationRate = 0.8

params.MutateWeightsProb = 0.8

params.WeightMutationMaxPower = 2.5
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.25

params.MaxWeight = 8

params.MutateAddNeuronProb = 0.1
params.MutateAddLinkProb = 0.1
params.MutateRemLinkProb = 0.0

params.MinActivationA  = 4.9
params.MaxActivationA  = 4.9

params.ActivationFunction_SignedSigmoid_Prob = 0.0
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0

params.CrossoverRate = 0.75  # mutate only 0.25
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2
class MultiNEATWrapper:
    """The NEATWrapper contains the means of initializing a NEAT
    network with a set of specific parameters, and the means of getting
    a Network at every timestep which represents the most fit (and
    currently tested) NEAT genome, and the means of updating the fitness
    of the NEAT genomes as they go."""
    def __init__(self, params, genome, population = None, 
                    *args, **kwargs):
        self.params = params
        if population is None:
            population = Population(genome, params, True, 2.0, 1)    
        self.population = population
        self.current = 0
        self.genomes = None


    def get_best_genome(self):
        return self.population.GetBestGenome()

    def get_current_genome(self, progress = True):
        print("GETTING CURRENT GENOME")
        if self.genomes is None:
            self.genomes = NEAT.GetGenomeList(self.population)
        if self.current >= len(self.genomes):
            self.update()
        genome = self.genomes[self.current]
        if progress: self.current += 1
        print("GETTING GENOME DONE")
        return genome
            
    def set_current_fitness(self, fit) -> None:
        self.get_current_genome().SetFitness(fit)

    def update(self) -> None:
        print("UPDATING POPULATION")
        self.population.Epoch()
        self.genomes = NEAT.GetGenomeList(self.population)
        self.current = 0
        print("UPDATING DONE")

    def copy(self): #returns a NEW OBJECT identical to current one
        Ellipsis

class MultiNEATController:
    def __init__(self, wrapper, steps = 100):
        self.wrapper = wrapper
        self.genome = None
        self.max_steps = steps
        self.fitness = 0
        self.step = 0
    def __call__(self, senses : np.array) -> np.array:
        if self.step == 0:
            self.genome = self.wrapper.get_current_genome()
        self.step += 1
        self.step %= self.max_steps 
        net = NEAT.NeuralNetwork()
        self.genome.BuildPhenotype(net)
        net.Input(senses)
        net.Activate()
        output = net.Output()
        o = []
        for i in range(len(output)):
            if output[i] > 0.7:
                o.append(1)
            elif output[i] < 0.3:
                o.append(-1)
            else:
                o.append(0)
        return o 

    def _feedback(self, *args): #TODO: Currently only takes a single bot's fitness
        if self.step == 0:
            self.fitness = args[0] 
            self.wrapper.set_current_fitness(self.fitness) #assumes 1-element feedback
            print("PREVIOUS FITNESS: ", self.fitness)
            self.fitness = 0

##



class TankDriveBot:
    def __init__(self, x, y, speed, rotate_rate, orientation : float, controller, los_range = 130):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.speed = speed
        self.rotate_rate = rotate_rate
        self.controller = controller
        self.rect = pygame.Rect(self.x,self.y,14,14)
        #self.surface = pygame.Surface((20, 25))
        #self.surface.fill(white)
        #self.surface.set_colorkey(blue)
        self.los_range = los_range 
        self.los = None

    def draw(self):
        #pygame.draw.ellipse(self.surface, black, self.surface.get_rect(), 2)
        pygame.draw.circle(screen, blue, (floor(self.x), floor(self.y)), 7, 1)
        self.los = pygame.draw.line(screen, black, (self.x, self.y), (self.x + sin(self.orientation) * self.los_range, 
                self.y + cos(self.orientation) * self.los_range))
        #screen.blit(self.surface, (self.x, self.y))

    def setpos(self,x,y):
        self.x = x
        self.y = y

    def move(self,l_wheel,r_wheel):
        tmp_x = 0
        tmp_y = 0
        tmp_x += self.x
        tmp_y += self.y
        if l_wheel == r_wheel and r_wheel != 0:
            sign = l_wheel / abs(l_wheel)
            self.x += sign * sin(self.orientation) * self.speed
            self.y += sign * cos(self.orientation) * self.speed

        elif l_wheel == 1 and r_wheel == -1: #pivot in-place towards the right
            self.orientation += self.rotate_rate    
        elif l_wheel == -1 and r_wheel == 1: #pivot in-place towards the left
            self.orientation -= self.rotate_rate    
        self.orientation %= (2 * pi) 
        self.rect = pygame.Rect(self.x,self.y,16,16)
        for c in collidables:
            if c != self:
                if self.rect.colliderect(c.rect):
                    self.x = tmp_x
                    self.y = tmp_y


    def rangefinder(self) -> float: 
        """Raycasts based on orientation / position of the bot, returns the distance from the closest object.
        
        For the moment performs a VERY costly computation of iterating over a distance."""
        distance = float('inf')
        if self.los is not None:
            for c in collidables:
                if c != self and self.los.colliderect(c.rect): #Collision detected
                    DIST_STEP = 1 
                    #Now we test a discrete set of distances from the origin (bot) to the end of the los
                    for i in np.arange(0, self.los_range, DIST_STEP): #iterate from 0 to range in DIST_STEP steps
                        point = (self.x + sin(self.orientation) * i, self.y + cos(self.orientation) * i)
                        if c.rect.collidepoint(point[0], point[1]):    
                            return float(i)
        return distance

         




if __name__ == '__main__':
    

    red = [255,0,0]
    green = [0,255,0]
    blue = [0,0,255]
    white = [255,255,255]
    black = [0,0,0]
    UP = [0,-1]
    DOWN = [0,1]
    LEFT = [-1,0]
    RIGHT = [1,0]
    NOTMOVING = [0,0]

    g = NEAT.Genome(0, 2, 0, 2, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, 0)
    NEAT_WRAPPER = MultiNEATWrapper(params, g, pop)
    #constants end
    #classes
    class collidable:
        x = 0
        y = 0
        w = 0
        h = 0
        rect = pygame.Rect(x,y,w,h)
        color = [0,0,0]
        def __init__(self,x,y,w,h,color):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.color = color
            self.rect = pygame.Rect(x,y,w,h)
        def draw(self):
            pygame.draw.rect(screen,self.color,[self.x,self.y,self.w,self.h],6)

    pygame.init()
    screenSize = [800,600]
    screenBGColor = white
    screen=pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SHIIIIIIIEEEEEEEEEEEEEEEEEEEEEEEEEET")
    player = TankDriveBot(screenSize[0]/2,screenSize[1]/2,3,0.2, 0, None)
    ##Initialize bots - CURRENTLY WITH DUMB CONTROLLERs
    bots = []
    NUM_BOTS = 2
    controller = MultiNEATController(NEAT_WRAPPER, 3000)
    for i in range(NUM_BOTS): #for now we just space them horizontally 
        #controller = DumbController()
        bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
    ##
    collidables = []
    clock=pygame.time.Clock()
    

    ##INITIALIZE OBSTACLES

    #INITIALIZE WALLS
    collidables.append(collidable(0, 0, 800, 3, blue))
    collidables.append(collidable(0, 0, 3, 600, blue))
    collidables.append(collidable(800, 0, 3, 600, blue))
    collidables.append(collidable(0, 600, 800, 3, blue))
    

    collidables.extend(bots)
    collidables.append(player)
    #INITIALIZE ACTUAL OBSTACLE


    #INITIALIZE WALLS
    ##

    running = True
    #globals end
    player_moving = NOTMOVING

    l_wheel = 0
    r_wheel = 0
    #functions
    def render():
        screen.fill(screenBGColor)
        clock.tick(400)
        for bot in bots:
            bot.draw()
        player.draw()
        for c in collidables:
            c.draw()
        pygame.display.flip()

    def reset():
        distance = 0
        for i in range(NUM_BOTS): #for now we just space them horizontally 
            bots[i].setpos(100 * i + 50, 50 * i + 50)
            distance += sqrt((bots[i].x - player.x) ** 2 + (bots[i].y - player.y) ** 2)
        bots[i].controller._feedback(distance)
        player.setpos(600 + 200 * random.random(), 100 + 300 * random.random())

    def tick():                                           #----------------HERE
        #for c in collidables:
        #    for d in collidables:
        #        if c != d:
        #            if d.rect.colliderect(c.rect):
        #                print("PHYSICAL COLLISION")
        for bot in bots:
            if bot.controller:
                ## MULTINEAT DISTANCE METRIC
                #if bot.rangefinder() != float('inf'):
                #    print(bot.rangefinder())
                if bot.controller.step == 0:
                    reset()
                distance = (bot.x - player.x) ** 2 + (bot.y - player.y) ** 2
                distance = sqrt(distance)
                ##
                print("BOT INPUT: ", [distance, bot.rangefinder()])
                instructions = bot.controller([distance, bot.rangefinder()]) #TODO: Add sensory inputs here
                bot.move(instructions[0], instructions[1])
        player.move(l_wheel, r_wheel)
        print("----------------------------------------------------") 
    #functions end
    
    #main loop
    while running==True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            l_wheel = 1
        elif keys[K_s]:
            l_wheel = -1
        else:
            l_wheel = 0
        if keys[K_UP]:
            r_wheel = 1
        elif keys[K_DOWN]:
            r_wheel = -1
        else:
            r_wheel = 0
        tick()
        render()
    #main loop end
    
    pygame.quit()
    




#class player:
#    x = 0
#    y = 0
#    speed = 0
#    rect = pygame.Rect(x,y,20,20)
#    def __init__(self,x,y,speed):
#        self.x = x
#        self.y = y
#        self.speed = speed
#        self.rect = pygame.Rect(self.x,self.y,20,20)
#    def draw(self):
#        if player_moving==LEFT:
#                pygame.draw.polygon(screen,black,[(self.x-10,self.y),(self.x+10,self.y-10),(self.x+10,self.y+10)])
#        elif player_moving==RIGHT:
#            pygame.draw.polygon(screen,black,[(self.x+10,self.y),(self.x-10,self.y-10),(self.x-10,self.y+10)])
#        elif player_moving==UP:
#            pygame.draw.polygon(screen,black,[(self.x,self.y-10),(self.x+10,self.y+10),(self.x-10,self.y+10)])
#        elif player_moving==DOWN:
#            pygame.draw.polygon(screen,black,[(self.x,self.y+10),(self.x+10,self.y-10),(self.x-10,self.y-10)])
#        else:
#            pygame.draw.rect(screen,black,pygame.Rect(self.x-10,self.y-10,20,20),6)
#    def setpos(self,x,y):
#        self.x = x
#        self.y = y
#    def move(self,direction):
#        self.x = self.x + direction[0]*self.speed
#        self.y = self.y + direction[1]*self.speed
#        self.rect = pygame.Rect(self.x,self.y,20,20)
