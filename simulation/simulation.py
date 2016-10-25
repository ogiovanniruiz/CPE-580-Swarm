import pygame
from pygame.locals import *
import random

import numpy as np
from math import sin, cos, pi, floor, sqrt
from abc import abstractmethod

import MultiNEAT as NEAT


## PERTINENT TODO -s listed here

    #Add penalty for having bots collide with objects (excluding LOS)
    #Wrap everything in objects, add parameters for bots to modify their "physics"
    #


##




class Controller:
    @abstractmethod
    def __call__(self, senses):
        """Maps a set of inputs (numpy array) to a set of outputs (numpy array)"""
        pass

    @abstractmethod
    def _feedback(self, *args):
        pass

class DumbController:
    def __call__(self, senses):
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
params.RecurrentProb = 0.15
params.OverallMutationRate = 0.8

params.MutateWeightsProb = 0.8

params.WeightMutationMaxPower = 2.5
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.25

params.MaxWeight = 8

params.MutateAddNeuronProb = 0.05
params.MutateAddLinkProb = 0.1
params.MutateRemLinkProb = 0.1

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
            
    def set_current_fitness(self, fit):
        self.get_current_genome().SetFitness(fit)

    def update(self):
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
    def __call__(self, senses):
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
    def __init__(self, x, y, speed, rotate_rate, orientation, controller, los_range = 130):
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

    def draw(self, screen):
        #pygame.draw.ellipse(self.surface, black, self.surface.get_rect(), 2)
        pygame.draw.circle(screen, blue, (int(floor(self.x)), int(floor(self.y))), 7, 1)
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


    def rangefinder(self): 
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

class Environment:
    def __init__(self, shape, controllers, bots, collidable, player = None):
        
        self.single_controller = (len(controllers) == 1) #should all bots operate on the same controller / GA?   
        self.controllers = controllers
        self.collidables = collidable
        self.bots = bots
        self.player = player
        self.shape = shape

        pygame.init()
        self.screenBGColor = white
        self.screen=pygame.display.set_mode(self.shape)
        pygame.display.set_caption("SHIIIIIIIEEEEEEEEEEEEEEEEEEEEEEEEEET")
        self.clock=pygame.time.Clock()


        self.running = True
        self.l_wheel = 0
        self.r_wheel = 0

    def play(self):
        while self.running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[K_w]:
                self.l_wheel = 1
            elif keys[K_s]:
                self.l_wheel = -1
            else:
                self.l_wheel = 0
            if keys[K_UP]:
                self.r_wheel = 1
            elif keys[K_DOWN]:
                self.r_wheel = -1
            else:
                self.r_wheel = 0
            self.tick()
            self.render()
        #main loop end
        
        pygame.quit()
    
    def render(self):
        self.screen.fill(self.screenBGColor)
        self.clock.tick(400)
        for bot in self.bots:
            bot.draw(self.screen)
        if self.player: self.player.draw(self.screen)
        for c in self.collidables:
            c.draw(self.screen)
        pygame.display.flip()

    def reset(self):
        distance = 0
        for i in range(len(self.bots)): #for now we just space them horizontally 
            self.bots[i].setpos(100 * i + 50, 50 * i + 50)
            distance += sqrt((self.bots[i].x - player.x) ** 2 + (self.bots[i].y - player.y) ** 2)
        self.bots[i].controller._feedback(distance)
        if self.player:
            self.player.setpos(600 + 200 * random.random(), 100 + 300 * random.random())

    def tick(self):                                           #----------------HERE
        #for c in collidables:
        #    for d in collidables:
        #        if c != d:
        #            if d.rect.colliderect(c.rect):
        #                print("PHYSICAL COLLISION")
        for bot in self.bots:
            if bot.controller:
                ## MULTINEAT DISTANCE METRIC
                #if bot.rangefinder() != float('inf'):
                #    print(bot.rangefinder())
                if bot.controller.step == 0:
                    self.reset()
                distance = (bot.x - self.player.x) ** 2 + (bot.y - self.player.y) ** 2
                distance = sqrt(distance)
                ##
                print("BOT INPUT: ", [distance, bot.rangefinder()])
                instructions = bot.controller([distance, bot.rangefinder()]) #TODO: Add sensory inputs here
                bot.move(instructions[0], instructions[1])
        self.player.move(self.l_wheel, self.r_wheel)
        print("----------------------------------------------------") 
    
if __name__ == '__main__':
    SCREENSIZE = [800, 600]


    g = NEAT.Genome(0, 2, 0, 2, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, 0)
    NEAT_WRAPPER = MultiNEATWrapper(params, g, pop)

    player = TankDriveBot(SCREENSIZE[0]/2,SCREENSIZE[1]/2,3,0.2, 0, None)
    bots = []
    NUM_BOTS = 2
    controller = MultiNEATController(NEAT_WRAPPER, 3000)
    for i in range(NUM_BOTS): #for now we just space them horizontally 
        #controller = DumbController()
        bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
    collidables = []

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
        def draw(self, screen):
            pygame.draw.rect(screen,self.color,[self.x,self.y,self.w,self.h],6)
    collidables.append(collidable(0, 0, 800, 3, blue))
    collidables.append(collidable(0, 0, 3, 600, blue))
    collidables.append(collidable(800, 0, 3, 600, blue))
    collidables.append(collidable(0, 600, 800, 3, blue))
    collidables.extend(bots)
    collidables.append(player)
    env = Environment(SCREENSIZE, [MultiNEATController(NEAT_WRAPPER, 3000)], bots, collidables, player = player)
    env.play()    
    
