import pygame
from pygame.locals import *
import random

import numpy as np
from math import sin, cos, pi, floor, sqrt
from abc import abstractmethod


#from keras.models import Sequential
#from keras.layers import Dense, Activation


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
        return np.array([random.choice((-1, 1)) for i in range(3)])
        return [0, 0]



## IF YOU DON'T HAVE MULTINEAT JUST COMMENT THIS OUT, DON'T REMOVE IT
import MultiNEAT as NEAT
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



class KerasWrapper:
    def __init__():
        pass


class KerasController: 
    pass 



class RangefinderBot:
    def __init__(self, x, y, speed, rotate_rate, orientation, controller, los_range = 130, rect_h = 16, rect_w = 16, color = blue):
        self.x = x
        self.y = y
        self.rect_h = rect_h
        self.rect_w = rect_w
        self.color = color
        self.orientation = orientation
        self.speed = speed
        self.rotate_rate = rotate_rate
        self.controller = controller
        self.rect = pygame.Rect(self.x,self.y,rect_h,rect_w)
        self.los_range = los_range 
        self.los = None
        
    def setpos(self,x,y):
        self.x = x
        self.y = y
    
    def draw(self, screen):
        #pygame.draw.ellipse(self.surface, black, self.surface.get_rect(), 2)
        if True: #to render rangefinder distance
            font = pygame.font.Font(None, 16)
            surface = font.render(str(self.rangefinder()), 0, blue)
            screen.blit(surface, (self.x, self.y))
        pygame.draw.circle(screen, self.color, (int(floor(self.x)), int(floor(self.y))), 7, 1)
        self.los = pygame.draw.line(screen, white, (self.x, self.y), (self.x + sin(self.orientation) * self.los_range, 
                self.y + cos(self.orientation) * self.los_range))
        #screen.blit(self.surface, (self.x, self.y))
    def rangefinder(self): 
        """Raycasts based on orientation / position of the bot, returns the distance from the closest object.
        
        For the moment performs a VERY costly computation of iterating over a distance."""
        color = [255, 255, 255]
        distance = float('inf')
        if self.los is not None:
            for c in collidables:
                if c != self and self.los.colliderect(c.rect): #Collision detected
                    color = c.color 
                    DIST_STEP = 1 
                    #Now we test a discrete set of distances from the origin (bot) to the end of the los
                    for i in np.arange(0, self.los_range, DIST_STEP): #iterate from 0 to range in DIST_STEP steps
                        point = (self.x + sin(self.orientation) * i, self.y + cos(self.orientation) * i)
                        if c.rect.collidepoint(point[0], point[1]):    
                            return float(i), color
        return distance, color 

class TankDriveBot(RangefinderBot):
    def move(self, l_wheel, r_wheel):
        stationary_flag = False
        collision_flag = False
        rotate_flag = False
        invalid_reverse_flag = False

        if l_wheel == r_wheel and r_wheel != 0:
            sign = l_wheel / abs(l_wheel)
            tmp_x = self.x + sign * sin(self.orientation) * self.speed
            tmp_y = self.y + sign * cos(self.orientation) * self.speed
            tmp_rect = pygame.Rect(tmp_x, tmp_y, 16, 16)
            for c in collidables:
                if tmp_rect.colliderect(c.rect) and c != self:
                    if hasattr(c, 'moveable') and c.moveable == True: #collided with moveable object
                        c.move(sign * sin(self.orientation) * self.speed, sign * cos(self.orientation) * self.speed)
                    collision_flag = True
            if not collision_flag:
                self.x = tmp_x
                self.y = tmp_y
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
        self.rect = pygame.Rect(self.x, self.y, 16, 16)
        self.orientation %= (2 * pi) 
        if self.stationary_time >= STATIONARY_THRESHOLD:
            stationary_flag = True
        if self.rotate_time * self.rotate_rate >= ROTATE_THRESHOLD:
            rotate_flag = True
        return (collision_flag, stationary_flag, rotate_flag, invalid_reverse_flag)

class LinearBot(RangefinderBot):
    def move(self, v, h, delta_orientation):
        self.orientation += delta_orientation * self.rotate_rate
        return (h * self.speed, v * self.speed)


class Environment:
    def __init__(self, shape, speed, controllers, bots, collidable, movement_func = None, feedback_func = None, penalty_func = None, player = []):
        self.speed = speed 
        #self.single_controller = (len(controllers) == 1) #should all bots operate on the same controller / GA?
        self.controllers = controllers
        self.collidables = collidable
        self.bots = bots
        self.player = player
        self.shape = shape

        pygame.init()
        self.screenBGColor = black
        self.screen=pygame.display.set_mode(self.shape)
        pygame.display.set_caption("CRAP")
        self.clock=pygame.time.Clock()
        self.running = True

        self.player_instructions = []


        self.feedback = 0 #the feedback whatever controller in the simulation will be operating on

        self.feedback_func = feedback_func
        self.movement_func = movement_func
        self.penalty_func = penalty_func
        ##Extraneous functionality
        pygame.font.init()

    def play(self):
        ## INITIALIZE POSITIONS
        positions = generate_positions_by_minimum_distance_with_obstacles(self.shape, len(self.bots), MINIMUM_BOT_DISTANCE)
        for i in range(len(self.bots)): #for now we just space them horizontally 
            #self.bots[i].setpos(100 * i + 50, 50 * i + 50)
            self.bots[i].setpos(positions[i][0], positions[i][1])
            self.bots[i].orientation = 0
        while self.running:
            self.player_instructions = []
            ## BEGIN GAME LOGIC
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            
            if keys[K_UP]:
                self.player_instructions.append(-1)
            elif keys[K_DOWN]:
                self.player_instructions.append(1)
            else:
                self.player_instructions.append(0)
            if keys[K_RIGHT]:
                self.player_instructions.append(1)
            elif keys[K_LEFT]:
                self.player_instructions.append(-1)
            else:
                self.player_instructions.append(0)
            if keys[K_w]:
                self.player_instructions.append(1) #orientation positive rotation
            elif keys[K_s]:
                self.player_instructions.append(-1)
            else:
                self.player_instructions.append(0)
            self.tick()
            self.render()
        #main loop end
        
        pygame.quit()
    
    def render(self):
        self.screen.fill(self.screenBGColor)
        self.clock.tick(self.speed)
        for c in self.collidables:
            c.draw(self.screen)
        pygame.display.flip()

    def reset(self):
        distance = 0
        #self.feedback = 0 #IGNORE PENALTIES IF THIS IS SET
        if self.feedback_func:
            self.feedback = self.feedback_func(self)
        for bot in self.bots:
            bot.controller._feedback(self.feedback)

        collidables = []
        collidables.append(collidable(0, 0, SCREENSIZE[0], 3, white))
        collidables.append(collidable(0, 0, 3, SCREENSIZE[1], white))
        collidables.append(collidable(SCREENSIZE[0] - 1, 0, 3, SCREENSIZE[1], white))
        collidables.append(collidable(0, SCREENSIZE[1] - 1, SCREENSIZE[0], 3, white))
        #initialize inner walls, if necessary
        #def initialize_collidable_obstacle(ENV_SHAPE, num, mean_width, w_var, mean_height, h_var, min_space = 100):
        collidable_initializer = initialize_collidable_obstacles(SCREENSIZE, 28, 100, 40, 3, 0, 70)
        collidables.extend(collidable_initializer) 
        collidable_initializer = initialize_collidable_obstacles(SCREENSIZE, 28, 3, 0, 110, 40, 70)
        collidables.extend(collidable_initializer) 
        #
        collidables.extend(bots)
        if player: collidables.append(player)
        print("GENERATION FEEDBACK: ", self.feedback)    
        self.feedback = 0 #reset feedback at the end of the epoch
        positions = generate_positions_by_minimum_distance_with_obstacles(self.shape, len(self.bots), MINIMUM_BOT_DISTANCE)
        while positions is None: #we try again <3
            positions = generate_positions_by_minimum_distance_with_obstacles(self.shape, len(self.bots), MINIMUM_BOT_DISTANCE)
        for i in range(len(self.bots)): #for now we just space them horizontally 
            #self.bots[i].setpos(100 * i + 50, 50 * i + 50)
            self.bots[i].setpos(positions[i][0], positions[i][1])
            self.bots[i].orientation = random.random() * 2 * pi 
        if len(self.player):
            pos = (600 + 200 * random.random(), 100 + 300 * random.random())
            for player in self.player:
                player.setpos(pos[0] + random.random * 10, pos[1] + random.random * 10)
                player.orientation = random.random() * 2 * pi

    def tick(self):                                           #----------------HERE
        collision = {}
        for bot in self.bots:
            if bot.controller:
                dist, color = bot.rangefinder() #gets distance and color from rangefinder, now
                instructions = bot.controller([bot.rangefinder(), color[0], color[1], color[2], 1.0]) #TODO: Add sensory inputs here
                collision = self.movement_func(self, bot, instructions, collision)
        if self.player:
            for player in self.player:
                collision = self.movement_func(self, player, self.player_instructions, collision)
        
        for c in range(len(collision)):
            net_x = sum([i[0] for i in collision.values()[c]])
            net_y = sum([i[1] for i in collision.values()[c]])
            print "NET_X: %x   NET_Y : %x" % (net_x, net_y)
            collision.keys()[c].move(net_x, net_y)

            


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

def generate_positions_by_minimum_distance_with_obstacles(shape, num_bots, min_distance):
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
        #CHECK FOR COLLISION / IN OBSTACLES
        collide = False
        for c in collidables:
            if c.rect.collidepoint(x, y):
                collide = True
        #
        if collide:
            continue
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

def initialize_collidable_obstacles(ENV_SHAPE, num, mean_width, w_var, mean_height, h_var, min_space = 100):
    collidables = []
    pos = generate_positions_by_minimum_distance_with_obstacles(ENV_SHAPE, num, min_space)
    for x, y in pos:
        height = mean_height + h_var * random.random() * random.choice((1, -1))
        width = mean_width + w_var * random.random() * random.choice((1, -1))
        collidables.append(collidable(x, y, height, width, white))
    return collidables

def move_bots(env, bot, instructions = [], collision = []):
    hit_wall = False
    res = bot.move(instructions[0], instructions[1], instructions[2])
    tmp_x = bot.x + res[0]
    tmp_y = bot.y + res[1]
    tmp_rect = pygame.Rect(tmp_x, tmp_y, bot.rect_h, bot.rect_w)
    for c in collidables:
        if tmp_rect.colliderect(c.rect) and c != bot:
            if hasattr(c, 'moveable') and c.moveable == True: #collided with moveable object
                collision.setdefault(c, []).append((res[0], res[1]))
            else: hit_wall = True
    if not len(collision) and not hit_wall:
        bot.x = tmp_x
        bot.y = tmp_y
    bot.rect = pygame.Rect(bot.x, bot.y, bot.rect_h, bot.rect_w)
    return collision

def feedback_by_relative_bot_distances(environment, *args):
        for i in range(len(self.bots)): #for now we just space them horizontally 
            closest = float('inf')
            for j in range(len(self.bots)): #TODO NOW WE MIMIZE DISTANCE TO CLOSEST BOT
                if j != i:
                    distance = sqrt((self.bots[i].x - self.bots[j].x) ** 2 + (self.bots[i].y - self.bots[j].y) ** 2)
                    if distance < closest:
                        closest = distance
            self.feedback += closest #should never be inf

def feedback_by_moveable_distance_to_region(environment, region): #region is a Rect 
    pass


def feedback_by_total_bot_distance_from_region(environment, region):  #region is a Rect
    pass


if __name__ == '__main__':
    SCREENSIZE = [1400, 750]
    COLLISION_PENALTY = 0.1 #penalize collisions, straight-up
    STATIONARY_PENALTY = 1 #penalize non-moving bots
    STATIONARY_THRESHOLD = 20 #time-threshold for immobile bots
    ROTATE_THRESHOLD = 4 #RADIANS
    ROTATE_PENALTY = 1

    INVALID_REVERSE_PENALTY = 2 #penalize moving in reverse without object within "threshold" of rangefinder
    INVALID_REVERSE_THRESHOLD = 15 #allowable distance at which to "allow" reverse

    DURATION = 6000 
    SPEED = 1000

    MINIMUM_BOT_DISTANCE = 300 #used for initializing the bots with random-but-spaced points

    players = []
    players.append(LinearBot(100, 100, 3, 0.2, 0, None))
    players.append(LinearBot(115, 115, 3, 0.2, 0, None))

    bots = []
    NUM_BOTS = 7
    controller = DumbController()

    for i in range(NUM_BOTS): #for now we just space them horizontally 

        #bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controllers[i]))
        bots.append(LinearBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
                        #TODO: Fix hitboxes on ALL objects to their visuals correspond to their actual hitboxes (instead of ... not)
    class collidable:     #TODO: CREDIT stackoverflow.com/questions/8195649/python-pygame-collision-detection-with-rects 
        def __init__(self,x,y,w,h,color):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.color = color
            self.rect = pygame.Rect(x,y,w,h)
            self.moveable = False
        def draw(self, screen):
            pygame.draw.rect(screen,self.color,[self.x,self.y,self.w,self.h],6)
        def move(self, d_X, d_Y): #move with force of delta X and delta Y
            pass

    class moveable(collidable):
        def __init__(self, inertia, *args):
            collidable.__init__(self, *args)
            self.moveable = True
            self.inertia = inertia

        def move(self, d_X, d_Y): #TODO: Add to seperate move cycle, to enable inertias with > single bot's potential delta
            factor = sqrt(d_X ** 2 + d_Y **2) / float(self.inertia)
            if factor > 1: #force overcomes inertia, move moveable
                collision_flag = False
                tmp_x = self.x + d_X * (1 / factor)
                tmp_y = self.y + d_Y * (1 / factor)
                tmp_rect = pygame.Rect(tmp_x, tmp_y, self.w, self.h)
                for c in collidables:
                    if tmp_rect.colliderect(c.rect) and c != self:
                        collision_flag = True
                if not collision_flag:
                    self.x = tmp_x
                    self.y = tmp_y
                    self.rect = tmp_rect



    #initialize outer walls
    collidables = []
    collidables.append(collidable(0, 0, SCREENSIZE[0], 3, blue))
    collidables.append(collidable(0, 0, 3, SCREENSIZE[1], blue))
    collidables.append(collidable(SCREENSIZE[0] - 1, 0, 3, SCREENSIZE[1], blue))
    collidables.append(collidable(0, SCREENSIZE[1] - 1, SCREENSIZE[0], 3, blue))
    #initialize inner walls, if necessary
    collidable_initializer = initialize_collidable_obstacles(SCREENSIZE, 28, 100, 40, 7, 0, 90)
    collidables.extend(collidable_initializer) 
    collidable_initializer = initialize_collidable_obstacles(SCREENSIZE, 28, 7, 0, 110, 40, 70)
    collidables.extend(collidable_initializer) 
    #
    collidables.extend(bots)
    collidables.extend([moveable(4.5, random.random() * 700, random.random() * 700, 30, 30, green) for i in range(7)])
    if len(players): collidables.extend(players)
    env = Environment(SCREENSIZE, SPEED, controller, bots, collidables, feedback_func = feedback_by_relative_bot_distances, 
            movement_func = move_bots, player = players)
    env.play() 



    
