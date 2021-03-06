import pygame
from pygame.locals import *
import random

import numpy as np
from math import sin, cos, pi, floor, sqrt
from abc import abstractmethod

import MultiNEAT as NEAT


## PERTINENT TODO -s listed here
    #Discret-ize the instructions at each timestep
    #VERIFY THAT MULTINEAT IS ACTUALLY TRAINING TO MINIMIZE THE FEEDBACK SIGNAL
    #FIX SPINNING BEHAVIOR GAHHHHHH PENALIZE IT
    #"Rank" distances so the largest distance any bot is from any others is penalized MORE (punish outliers)
    #Penalize stagnation ONLY if distance is above some threshold? 
    #Add BETTER control for bots so they can move AND turn at once 
    #Add more realistic physics for bots
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
        if self.genomes is None:
            self.genomes = NEAT.GetGenomeList(self.population)
        if self.current >= len(self.genomes):
            self.update()
        genome = self.genomes[self.current]
        if progress: self.current += 1
        return genome
            
    def set_current_fitness(self, fit):
        self.get_current_genome().SetFitness(fit)

    def update(self):
        print " ------------------------------ CURRENT BEST FITNESS: ", self.population.GetBestGenome().GetFitness()
        print " ------------------------------ BEST FITNESS EVER: ", self.population.GetBestFitnessEver()
        self.population.Epoch()
        self.genomes = NEAT.GetGenomeList(self.population)
        self.current = 0

    def copy(self): #returns a NEW OBJECT identical to current one
        Ellipsis

class MultiNEATController:
    def __init__(self, wrapper, num_bots, steps = 100):
        self.wrapper = wrapper
        self.genome = None
        self.networks = {} 
        for i in range(num_bots):
            self.networks[i] = NEAT.NeuralNetwork()
        self.max_steps = steps
        self.fitness = 0
        self.step = 0

    def __call__(self, senses, bot_id):
        if self.step == 0:
            self.genome = self.wrapper.get_current_genome()
            for net in self.networks.values():
                self.genome.BuildPhenotype(net) #theoretically build network 
        net = self.networks[bot_id]
        self.step += 1
        self.step %= self.max_steps 
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
        self.tick()
        return o 

    def tick(self):
        self.step += 1
        self.step %= self.max_steps

    def _feedback(self, *args): #TODO: Currently only takes a single bot's fitness
        if self.step == 0:
            self.fitness = args[0] 
            self.wrapper.set_current_fitness(self.fitness) #assumes 1-element feedback
            self.fitness = 0

##
class 



##


class TankDriveBot:
    def __init__(self, x, y, speed, rotate_rate, orientation, controller, los_range = 130):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.speed = speed
        self.rotate_rate = rotate_rate
        self.controller = controller
        self.rect = pygame.Rect(self.x,self.y,16,16)
        #self.surface = pygame.Surface((20, 25))
        #self.surface.fill(white)
        #self.surface.set_colorkey(blue)
        self.los_range = los_range 
        self.los = None
        
        self.stationary_time = 0
        self.rotate_time = 0
        self.rotate_direction = 0

    def draw(self, screen):
        #pygame.draw.ellipse(self.surface, black, self.surface.get_rect(), 2)
        if True: #to render rangefinder distance
            font = pygame.font.Font(None, 16)
            surface = font.render(str(self.rangefinder()), 0, blue)
            screen.blit(surface, (self.x, self.y))
        pygame.draw.circle(screen, blue, (int(floor(self.x)), int(floor(self.y))), 7, 1)
        self.los = pygame.draw.line(screen, black, (self.x, self.y), (self.x + sin(self.orientation) * self.los_range, 
                self.y + cos(self.orientation) * self.los_range))
        #screen.blit(self.surface, (self.x, self.y))

    def setpos(self,x,y):
        self.x = x
        self.y = y

    def move(self,l_wheel,r_wheel):
        stationary_flag = False
        collision_flag = False
        rotate_flag = False
        invalid_reverse_flag = False
        tmp_x = 0
        tmp_y = 0
        tmp_x += self.x
        tmp_y += self.y

        if l_wheel == r_wheel and r_wheel != 0:
            sign = l_wheel / abs(l_wheel)
            self.x += sign * sin(self.orientation) * self.speed
            self.y += sign * cos(self.orientation) * self.speed
            if sign < 0: #moving in REVERSE
                if self.rangefinder() > INVALID_REVERSE_THRESHOLD:
                    invalid_reverse_flag = True
            self.rotate_time = 0
        elif l_wheel == 1 and r_wheel == -1: #pivot in-place towards the right
            self.orientation += self.rotate_rate    
            if self.rotate_direction != 1:
                self.rotate_direction = 1
                self.rotate_time = 0
            else:
                self.rotate_time += 1
        elif l_wheel == -1 and r_wheel == 1: #pivot in-place towards the left
            self.orientation -= self.rotate_rate    
            if self.rotate_direction != -1:
                self.rotate_direction = -1
                self.rotate_time = 0
            else:
                self.rotate_time += 1
        self.orientation %= (2 * pi) 
        if tmp_x == self.x and tmp_y == self.y:
            self.stationary_time += 1
        else:
            self.stationary_time = 0
            for c in collidables:
                if c != self and c != self.los:
                    if self.rect.colliderect(c.rect):
                        self.x = tmp_x
                        self.y = tmp_y
                        collision_flag = True
            self.rect = pygame.Rect(self.x,self.y,16,16)
        if self.stationary_time >= STATIONARY_THRESHOLD:
            stationary_flag = True
        if self.rotate_time * self.rotate_rate >= ROTATE_THRESHOLD:
            rotate_flag = True
        return (collision_flag, stationary_flag, rotate_flag, invalid_reverse_flag)
            


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

def generate_positions_by_minimum_distance(shape, num_bots, min_distance):
    """Returns a list of positions for all the bots such that each bot is some minimum distance
    from all the other bots, but their positions are otherwise randomly generated.
    
    Positions are returned as (x, y) tuples."""
    x_pos = []
    y_pos = []
    
    x_points = list(np.arange(10, shape[0] - 10))
    y_points = list(np.arange(10, shape[1] - 10))
    while len(x_pos) < num_bots and len(x_points) > 0 and len(y_points) > 0:
        ind_x = random.choice(range(len(x_points))) #select random point
        ind_y = random.choice(range(len(y_points)))
        x = x_points.pop(ind_x)
        y = y_points.pop(ind_y)
        minimum = float('inf')  #keep track of minimum distance
        for point in range(len(x_pos)):
            distance = sqrt((x_pos[point] - x)**2 + (y_pos[point] - y)**2)        
            if distance < minimum:
                minimum = distance
        if minimum <= min_distance:
            continue
        else:
            x_pos.append(x)
            y_pos.append(y)
    if len(x_pos) < num_bots:
        print "SHIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIi", zip(x_pos, y_pos)
        return None
    else:
        #print "Remaining points: %s %s" % (len(x_points), len(y_points))
        return zip(x_pos, y_pos)



class Environment:
    def __init__(self, shape, speed, controllers, bots, collidable, player = None):
        self.speed = speed 
        self.single_controller = (len(controllers) == 1) #should all bots operate on the same controller / GA?   
        self.controllers = controllers
        self.collidables = collidable
        self.bots = bots
        self.player = player
        self.shape = shape

        pygame.init()
        self.screenBGColor = white
        self.screen=pygame.display.set_mode(self.shape)
        pygame.display.set_caption("CRAP")
        self.clock=pygame.time.Clock()


        self.running = True
        self.l_wheel = 0
        self.r_wheel = 0


        self.feedback = 0 #the feedback whatever controller in the simulation will be operating on


        ##Extraneous functionality
        pygame.font.init()

    def play(self):
        ## INITIALIZE POSITIONS
        positions = generate_positions_by_minimum_distance(self.shape, len(self.bots), MINIMUM_BOT_DISTANCE)
        for i in range(len(self.bots)): #for now we just space them horizontally 
            #self.bots[i].setpos(100 * i + 50, 50 * i + 50)
            self.bots[i].setpos(positions[i][0], positions[i][1])
            self.bots[i].orientation = 0
        if self.player:
            self.player.setpos(600 + 200 * random.random(), 100 + 300 * random.random())
            self.player.orientation = 0
        while self.running:
            ## BEGIN GAME LOGIC
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
        self.clock.tick(self.speed)
        for bot in self.bots:
            bot.draw(self.screen)
        if self.player: self.player.draw(self.screen)
        for c in self.collidables:
            c.draw(self.screen)
        pygame.display.flip()

    def reset(self):
        distance = 0
        #self.feedback = 0 #IGNORE PENALTIES IF THIS IS SET
        for i in range(len(self.bots)): #for now we just space them horizontally 
            closest = float('inf')
            for j in range(len(self.bots)): #TODO NOW WE MIMIZE DISTANCE TO CLOSEST BOT
                if j != i:
                    distance = sqrt((self.bots[i].x - self.bots[j].x) ** 2 + (self.bots[i].y - self.bots[j].y) ** 2)
                    if distance < closest:
                        closest = distance
            self.feedback += closest #should never be inf
        self.bots[i].controller._feedback(-1 * self.feedback)
        #self.bots[i].controller._feedback(self.feedback)
        print "GENERATION FEEDBACK: ", -1 * self.feedback    
        #print "GENERATION FEEDBACK: ", self.feedback    
        self.feedback = 0 #reset feedback at the end of the epoch
        positions = generate_positions_by_minimum_distance(self.shape, len(self.bots), MINIMUM_BOT_DISTANCE)
        while positions is None: #we try again <3
            positions = generate_positions_by_minimum_distance(self.shape, len(self.bots), MINIMUM_BOT_DISTANCE)
        for i in range(len(self.bots)): #for now we just space them horizontally 
            #self.bots[i].setpos(100 * i + 50, 50 * i + 50)
            self.bots[i].setpos(positions[i][0], positions[i][1])
            self.bots[i].orientation = random.random() * 2 * pi 
        if self.player:
            self.player.setpos(600 + 200 * random.random(), 100 + 300 * random.random())
            self.player.orientation = random.random() * 2 * pi

    def tick(self):                                           #----------------HERE
        #for c in collidables:
        #    for d in collidables:
        #        if c != d:
        #            if d.rect.colliderect(c.rect):
        #                print("PHYSICAL COLLISION")
        collisions = 0
        for i in range(len(self.bots)):
        #for bot in self.bots:
            if self.bots[i].controller:
                ## MULTINEAT DISTANCE METRIC
                #if bot.rangefinder() != float('inf'):
                #    print(bot.rangefinder())
                if self.bots[i].controller.step == 0:
                    self.reset()
                instructions = self.bots[i].controller([self.bots[i].rangefinder(), 1.0], i) #TODO: Add sensory inputs here
                res = self.bots[i].move(instructions[0], instructions[1])
                if res is not None:
                    if res[0]: #collision flag
                        self.feedback += COLLISION_PENALTY
                    if res[1]: #stagnant_flag
                        self.feedback += STATIONARY_PENALTY 
                    if res[2]: #rotate flag
                        self.feedback += ROTATE_PENALTY
                    if res[3]: #invalid_reverse flag
                        self.feedback += INVALID_REVERSE_PENALTY
        if self.player:
            res = self.player.move(self.l_wheel, self.r_wheel)
            if res[0]: #collision flag
                self.feedback += COLLISION_PENALTY
            if res[1]: #stagnant_flag
                self.feedback += STATIONARY_PENALTY 



if __name__ == '__main__':
    SCREENSIZE = [800, 600]
    COLLISION_PENALTY = 0.1 #penalize collisions, straight-up
    STATIONARY_PENALTY = 2 #penalize non-moving bots
    STATIONARY_THRESHOLD = 40 #time-threshold for immobile bots
    ROTATE_THRESHOLD = 4 #RADIANS
    ROTATE_PENALTY = 2

    INVALID_REVERSE_PENALTY = 3 #penalize moving in reverse without object within "threshold" of rangefinder
    INVALID_REVERSE_THRESHOLD = 40 #allowable distance at which to "allow" reverse

    DURATION = 9000 
    SPEED = 1000

    MINIMUM_BOT_DISTANCE = 300 #used for initializing the bots with random-but-spaced points

    NUM_BOTS = 5
    BOT_LOS = 200


    params = NEAT.Parameters()
    params.PopulationSize = 60
    params.DynamicCompatibility = True
    params.WeightDiffCoeff = 4.0
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 10
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 20
    params.MinSpecies = 3
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.1
    params.OverallMutationRate = 0.6
    
    params.MutateWeightsProb = 0.2
    
    params.WeightMutationMaxPower = 2.5
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.25
    
    params.MaxWeight = 11
    
    params.MutateAddNeuronProb = 0.1
    params.MutateAddLinkProb = 0.1
    params.MutateRemLinkProb = 0.1
    
    params.MinActivationA  = 4.9
    params.MaxActivationA  = 4.9
    
    params.ActivationFunction_SignedSigmoid_Prob = 0.15
    #params.ActivationFunction_UnsignedSigmoid_Prob = 0.45
    params.ActivationFunction_Linear_Prob = 0.9
    params.ActivationFunction_Tanh_Prob = 0.2
    params.ActivationFunction_SignedStep_Prob = 0.2
    
    params.CrossoverRate = 0.25  # mutate only 0.25
    params.MultipointCrossoverRate = 0.25
    params.SurvivalRate = 0.2
    g = NEAT.Genome(0, 2, 0, 2, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, 0)
    NEAT_WRAPPER = MultiNEATWrapper(params, g, pop)

    #player = TankDriveBot(SCREENSIZE[0]/2,SCREENSIZE[1]/2,2,0.05, 0, None)
    player = None
    bots = []
    controller = MultiNEATController(NEAT_WRAPPER, NUM_BOTS, DURATION)
    #controllers = [MultiNEATController(NEAT_WRAPPER, DURATION) for i in range(NUM_BOTS)]
    for i in range(NUM_BOTS): #for now we just space them horizontally 
        #controller = DumbController()
        #bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controllers[i]))
        bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller, BOT_LOS))
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
    collidables.append(collidable(799, 0, 3, 600, blue))
    collidables.append(collidable(0, 599, 800, 3, blue))
    collidables.extend(bots)
    if player: collidables.append(player)
    env = Environment(SCREENSIZE, SPEED, [1, 2, 3], bots, collidables, player = player)
    env.play()    
    
