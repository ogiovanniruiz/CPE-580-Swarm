import pygame
from pygame.locals import *
import random

import numpy as np
from math import sin, cos, pi, floor, sqrt
from abc import abstractmethod


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

MAZE = True
COLLECTION = not MAZE

MOVEABLE_FRICTION = 0.4
BOT_MOMENTUM = 0.3
MAX_SPEED = sqrt(18.0)
class Collidable:     #TODO: CREDIT stackoverflow.com/questions/8195649/python-pygame-collision-detection-with-rects 
    def __init__(self,x,y,w,h,color):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.rect = pygame.Rect(x,y,w,h)
        self.moveable = False
    def draw(self, env, screen):
        pygame.draw.rect(screen,self.color,[self.x,self.y,self.w,self.h],6)
    def move(self, d_X, d_Y): #move with force of delta X and delta Y
        pass

class Moveable(Collidable):
    def __init__(self, inertia, *args):
        Collidable.__init__(self, *args)
        self.moveable = True
        self.inertia = inertia

    def move(self, env, d_X, d_Y, collision): #TODO: Add to seperate move cycle, to enable inertias with > single bot's potential delta
        global MOVEABLE_FRICTION 
        factor = sqrt(d_X ** 2 + d_Y **2) / float(self.inertia)
        if factor > 1: #force overcomes inertia, move moveable
            dim_X = d_X * (1 / float(factor))
            dim_Y = d_Y * (1 / float(factor))
            collision_flag = False
            tmp_x = self.x + dim_X
            tmp_y = self.y + dim_Y 
            tmp_rect = pygame.Rect(tmp_x, tmp_y, self.w, self.h)
            for c in env.collidables:
                if tmp_rect.colliderect(c.rect) and c != self:
                    if hasattr(c, 'moveable') and c.moveable == True:
                        dim_X *= MOVEABLE_FRICTION
                        dim_Y *= MOVEABLE_FRICTION
                        collision.setdefault(c, []).append((dim_X, dim_Y))
                        tmp_x -= dim_X 
                        tmp_y -= dim_Y
                    else:
                        collision_flag = True
            if not collision_flag:
                self.x = tmp_x
                self.y = tmp_y
                self.rect = tmp_rect
        return collision
    
                #if hasattr(c, 'moveable') and c.moveable == True: #collided with moveable object
                #    collision.setdefault(c, []).append((res[0], res[1]))

class Controller:
    @abstractmethod
    def __call__(self, senses, bot_id):
        """Maps a set of inputs (numpy array) to a set of outputs (numpy array)"""
        pass

    @abstractmethod
    def _feedback(self, terminal = False, *args):
        pass

class DumbController(Controller):
    def __call__(self, senses, bot_id):
        return np.array([random.choice((-1, 1)) for i in range(3)])
        return [0, 0]

    def _feedback(self, terminal = False, *args):
        pass




## IF YOU DON'T HAVE MULTINEAT JUST COMMENT THIS OUT, DON'T REMOVE IT
try:
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
        def __init__(self, wrapper, num_bots):
            self.wrapper = wrapper
            self.genome = None 
            self.networks = {} 
            for i in range(num_bots):
                self.networks[i] = NEAT.NeuralNetwork()
            self.fitness = 0
            self.steps = 0 #for keeping track of terminal feedback calls
    
            #NECESSARY TO INITIALIZE GENOMES, I KNOW THIS SHOULDN"T BE INIT STEP
            self.wrapper.set_current_fitness(self.fitness) #assumes 1-element feedback
    
            self.genome = self.wrapper.get_current_genome()
            for net in self.networks.values():
               self.genome.BuildPhenotype(net) #theoretically build network 
    
    
        def __call__(self, senses, bot_id):
            tmp = senses[:]
            senses = []
            MAX_OUTPUT = 100000
            for s in tmp:
                if s > MAX_OUTPUT: s = MAX_OUTPUT
                senses.append(s)
    
            net = self.networks[bot_id]
            net.Input(senses)
            net.Activate()
            output = net.Output()
            #print "OUTPUT: ", [o for o in output]
            self.tick()
            #self.steps += 1
            return output 
    
        def tick(self): #if "stpes' need to be reintroduced
            self.steps += 1
    
        def _feedback(self, terminal = False, *args): #TODO: Currently only takes a single bot's fitness
            if terminal == True and self.steps > 0: #only update / change genomes on update = True
                #self.fitness = -1 * args[0] 
                self.fitness = args[0]
                self.steps = 0
                self.wrapper.set_current_fitness(self.fitness) #assumes 1-element feedback
    
                self.genome = self.wrapper.get_current_genome()
                for net in self.networks.values():
                   self.genome.BuildPhenotype(net) #theoretically build network 
    
    
    params = NEAT.Parameters()
    params.PopulationSize = 30
    params.DynamicCompatibility = True
    params.WeightDiffCoeff = 4.0
    params.CompatTreshold = 1.5
    params.YoungAgeTreshold = 12
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 20
    params.MinSpecies = 1
    params.MaxSpecies = 5
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.15
    params.OverallMutationRate = 0.3
        
    params.MutateWeightsProb = 0.2
        
    params.WeightMutationMaxPower = 2.5
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.4
    params.WeightMutationRate = 0.45
        
    params.MaxWeight = 9
        
    params.MutateAddNeuronProb = 0.1
    params.MutateAddLinkProb = 0.1
    params.MutateRemLinkProb = 0.1
        
    params.MinActivationA  = 4.9
    params.MaxActivationA  = 4.9
        
    params.ActivationFunction_SignedSigmoid_Prob = 0.35
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.15
    params.ActivationFunction_Linear_Prob = 0.1
    params.ActivationFunction_Tanh_Prob = 0.3
    params.ActivationFunction_SignedStep_Prob = 0.2
    
    params.CrossoverRate = 0.15  # mutate only 0.25
    params.MultipointCrossoverRate = 0.25
    params.SurvivalRate = 0.2
    
except:
    pass
    
    #g = NEAT.Genome(0, 6, 0, 3, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)
    g = NEAT.Genome(0, 29, 0, 3, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, 0)
    NEAT_WRAPPER = MultiNEATWrapper(params, g, pop)
    
##



class RangefinderBot:
    def __init__(self, x, y, speed, rotate_rate, orientation, controller, los_range = 90, rect_h = 16, rect_w = 16, color = blue):
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

        self.vector = np.zeros(2)
        
    def setpos(self,x,y):
        self.x = x
        self.y = y
    
    def draw(self, env, screen):
        #pygame.draw.ellipse(self.surface, black, self.surface.get_rect(), 2)
        if True: #to render rangefinder distance
            font = pygame.font.Font(None, 16)
            try:
                surface = font.render(str(self.rangefinder(env)), 0, blue)
                screen.blit(surface, (self.x, self.y))
            except:  #out of range
                pass
        pygame.draw.circle(screen, self.color, (int(floor(self.x)), int(floor(self.y))), 7, 1)
        try:
            self.los = pygame.draw.line(screen, white, (self.x, self.y), (self.x + sin(self.orientation) * self.los_range, 
                    self.y + cos(self.orientation) * self.los_range))
        except: #out of range
            pass
        #screen.blit(self.surface, (self.x, self.y))
    def rangefinder(self, env): 
        """Raycasts based on orientation / position of the bot, returns the distance from the closest object.
        
        For the moment performs a VERY costly computation of iterating over a distance."""
        color = [255, 255, 255]
        distance = float('inf')
        collisions = [] #TODO: Utilize this to fix multiple objects in LOS not being prioritized by distance (min)
        if self.los is not None:
            if self.los.colliderect(env.default_region):
                color = red
                DIST_STEP = 1 
                #Now we test a discrete set of distances from the origin (bot) to the end of the los
                for i in np.arange(0, self.los_range, DIST_STEP): #iterate from 0 to range in DIST_STEP steps
                    point = (self.x + sin(self.orientation) * i, self.y + cos(self.orientation) * i)
                    if env.default_region.collidepoint(point[0], point[1]):    
                        return float(i), color

            for c in env.collidables:
                if c != self and self.los.colliderect(c.rect): #Collision detected
                    try:
                        collisions.append(c)
                        color = c.color 
                        DIST_STEP = 1 
                        #Now we test a discrete set of distances from the origin (bot) to the end of the los
                        for i in np.arange(0, self.los_range, DIST_STEP): #iterate from 0 to range in DIST_STEP steps
                            point = (self.x + sin(self.orientation) * i, self.y + cos(self.orientation) * i)
                            if c.rect.collidepoint(point[0], point[1]):    
                                return float(i), color
                    except:
                        pass
        return distance, color 

class PrickleBot(RangefinderBot):
    def __init__(self, num_los = 4, *args, **kwargs): 
        RangefinderBot.__init__(self, *args, **kwargs) 
        self.num_los = num_los
        self.los_inc = 2. * pi / num_los    #spacing between los lines

    
    def draw(self, env, screen):
        #pygame.draw.ellipse(self.surface, black, self.surface.get_rect(), 2)
        if True: #to render rangefinder distance
            font = pygame.font.Font(None, 16)
            try:
                surface = font.render(str(self.rangefinder(env)), 0, blue)
                screen.blit(surface, (self.x, self.y))
            except:  #out of range
                pass
        pygame.draw.circle(screen, self.color, (int(floor(self.x)), int(floor(self.y))), 7, 1)
        self.los = []
        for l in range(self.num_los):
            try:
                self.los.append(pygame.draw.line(screen, white, (self.x, self.y), (self.x + sin(self.los_inc * l) * self.los_range, 
                        self.y + cos(self.los_inc * l) * self.los_range)))
            except: #out of range
                self.los.append(None)
            #screen.blit(self.surface, (self.x, self.y))


    def rangefinder(self, env):
        colors = []
        distances = []
        los = None
        collisions = [] #TODO: Utilize this to fix multiple objects in LOS not being prioritized by distance (min)
        for l in range(self.num_los):
            collision_point = float('inf')
            color = [255, 255, 255]
            try:
                los = self.los[l]
            except TypeError:
                los = None
            if los is not None:
                if los.colliderect(env.default_region):
                    color = red
                    DIST_STEP = 1 
                    #Now we test a discrete set of distances from the origin (bot) to the end of the los
                    for i in np.arange(0, self.los_range, DIST_STEP): #iterate from 0 to range in DIST_STEP steps
                        if collision_point < self.los_range:
                            break
                        point = (self.x + sin(self.orientation) * i, self.y + cos(self.orientation) * i)
                        if env.default_region.collidepoint(point[0], point[1]):    
                            collision_point = i
                
                for c in env.collidables:
                    if collision_point < self.los_range:
                        break
                    if c != self and los.colliderect(c.rect): #Collision detected
                        try:
                            collisions.append(c)
                            color = c.color 
                            DIST_STEP = 1 
                            #Now we test a discrete set of distances from the origin (bot) to the end of the los
                            for i in np.arange(0, self.los_range, DIST_STEP): #iterate from 0 to range in DIST_STEP steps
                                point = (self.x + sin(self.orientation) * i, self.y + cos(self.orientation) * i)
                                if c.rect.collidepoint(point[0], point[1]):    
                                    collision_point = i
                        except:
                            pass
                colors.append(color)
                distances.append(collision_point)
        return distances, colors 


class TankDriveBot(RangefinderBot):
    def move(self, env, l_wheel, r_wheel):
        stationary_flag = False
        collision_flag = False
        rotate_flag = False
        invalid_reverse_flag = False

        if l_wheel == r_wheel and r_wheel != 0:
            sign = l_wheel / abs(l_wheel)
            tmp_x = self.x + sign * sin(self.orientation) * self.speed
            tmp_y = self.y + sign * cos(self.orientation) * self.speed
            tmp_rect = pygame.Rect(tmp_x, tmp_y, 16, 16)
            for c in env.collidables:
                if tmp_rect.colliderect(c.rect) and c != self:
                    if hasattr(c, 'moveable') and c.moveable == True: #collided with moveable object
                        c.move(sign * sin(self.orientation) * self.speed, sign * cos(self.orientation) * self.speed)
                    collision_flag = True
            if not collision_flag:
                self.x = tmp_x
                self.y = tmp_y
            if sign < 0: #moving in REVERSE
                if self.rangefinder(environment) > INVALID_REVERSE_THRESHOLD:
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

class LinearRangefinderBot(RangefinderBot):
    def move(self, v, h, delta_orientation):
        global BOT_MOMENTUM, MAX_SPEED
        self.orientation += delta_orientation * self.rotate_rate
        self.orientation %= (2 * pi) 
        speed = [self.vector[0] + h * self.speed, self.vector[1] + v * self.speed]#v * self.speed)
        magnitude = sqrt(abs(speed[0]) + abs(speed[1]))
        if magnitude > MAX_SPEED:
            factor = MAX_SPEED / float(magnitude)
            speed[0] *= factor
            speed[1] *= factor
        self.vector = np.array((BOT_MOMENTUM * speed[0], BOT_MOMENTUM * speed[1]))
        return speed

class LinearPrickleBot(PrickleBot):
    def move(self, v, h, delta_orientation):
        global BOT_MOMENTUM, MAX_SPEED
        self.orientation += delta_orientation * self.rotate_rate
        self.orientation %= (2 * pi) 
        speed = [self.vector[0] + h * self.speed, self.vector[1] + v * self.speed]#v * self.speed)
        magnitude = sqrt(abs(speed[0]) + abs(speed[1]))
        if magnitude > MAX_SPEED:
            factor = MAX_SPEED / float(magnitude)
            speed[0] *= factor
            speed[1] *= factor
        self.vector = np.array((BOT_MOMENTUM * speed[0], BOT_MOMENTUM * speed[1]))
        return speed



class Environment:
    def __init__(self, shape, speed, controllers, bots, movement_func = None, sensor_func = None, 
            feedback_func = None, reset_func = None, penalty_func = None, player = [], default_region = None, 
            default_clock_threshold = 9000):
        self.speed = speed 
        #self.single_controller = (len(controllers) == 1) #should all bots operate on the same controller / GA?
        self.controllers = controllers
        self.bots = bots
        self.collidables = []
        self.player = player
        self.shape = shape

        pygame.init()
        self.screenBGColor = black
        self.screen=pygame.display.set_mode(self.shape)
        pygame.display.set_caption("MAZE RUNNER")
        self.clock=pygame.time.Clock()
        self.step = 0
        self.running = True

        self.player_instructions = []

        self.default_region = default_region
        self.default_clock_threshold = default_clock_threshold
        self.iteration = 0
        self.feedback = 0 #the feedback whatever controller in the simulation will be operating on

        self.feedback_func = feedback_func
        self.movement_func = movement_func
        self.penalty_func = penalty_func
        self.reset_func = reset_func
        self.sensor_func = sensor_func
        ##Extraneous functionality
        pygame.font.init()

    def play(self):
        ## INITIALIZE POSITIONS
        self.reset(True)
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
            c.draw(self, self.screen)
        if self.default_region:
            pygame.draw.rect(self.screen, red, self.default_region, 1)
        pygame.display.flip()

    def reset(self, initial = False):
        self.step = 0
        distance = 0
        #self.feedback = 0 #IGNORE PENALTIES IF THIS IS SET
        if self.feedback_func is not None:
            self.feedback = self.feedback_func(self, self.default_region)
            print "STEP %s Feedback : %s" % (self.step, self.feedback)
            _id = 0
            for bot in self.bots:
                bot.controller._feedback(True, self.feedback, _id) #if shared controller this SHOULDN'T BREAK ANYTHING
                _id += 1
        self.iteration += 1

        collidables = []
        collidables.append(Collidable(5, 5, self.shape[0], 3, blue))
        collidables.append(Collidable(5, 5, 3, self.shape[1], blue))
        collidables.append(Collidable(self.shape[0] - 5, 5, 3, self.shape[1], blue))
        collidables.append(Collidable(5, self.shape[1] - 5, self.shape[0], 3, blue))
        if MAZE == True: 
            ##initialize inner walls, if necessary
            collidable_initializer = initialize_collidable_obstacles(self, self.shape, 20, 100, 50, 7, 0, 90)
            collidables.extend(collidable_initializer) 
            collidable_initializer = initialize_collidable_obstacles(self, self.shape, 20, 7, 0, 100, 50, 100)
            collidables.extend(collidable_initializer) 
            #
        elif COLLECTION == True: 
            #initialize moveables!
            moveables = []
            moveables = [Moveable(1.5, self.shape[0] / 2 + 100 + i * 20, self.shape[1] / 2 + i * 50, 15, 15, green) for i in range(3)]                 
            moveables.extend([Moveable(1.5, self.shape[0] / 2 + i * 20, self.shape[1] / 2 + i * 50, 15, 15, green) for i in range(3)])
            moveables = [Moveable(1.5, 
                self.shape[0]/2 + random.random() * 100, self.shape[1]/2 + random.choice((-1, 1)) * random.random() * 200 , 
                15, 15, green) for i in range(5)]
            #moveables = [Moveable(1.5, self.shape[0] / 2 + 100 + i * 20, self.shape[1] / 2 + i * 50, 15, 15, green) for i in range(4)]                 
            collidables.extend(moveables)

        collidables.extend(self.bots)
        self.collidables = collidables 
        ## Reset objects here!
        positions = None
        while positions is None:
            #positions = generate_positions_by_minimum_distance_with_obstacles(self, self.shape, len(self.bots), MINIMUM_BOT_DISTANCE)
            positions = generate_positions_within_region_with_obstacles(self, self.shape, len(self.bots), 
                    pygame.Rect(self.shape[0] - 200, self.shape[1] - 200, 300, 300))
        for i in range(len(self.bots)): #for now we just space them horizontally 
            #self.bots[i].setpos(100 * i + 50, 50 * i + 50)
            self.bots[i].setpos(positions[i][0], positions[i][1])
            self.bots[i].orientation = pi 

        players = []
        #players.append(LinearRangefinderBot(100, 100, 3, 0.2, 0, None))
        #players.append(LinearBot(115, 115, 3, 0.2, 0, None))


        if len(players): 
            collidables.extend(players)
            self.player = players
        self.collidables = collidables 

    def tick(self):                                           #----------------HERE
        if self.reset_func(self, self.default_clock_threshold):
            self.reset()
        else:
            if self.feedback_func is not None:
                self.feedback = self.feedback_func(self, self.default_region)
                print "STEP %s Feedback : %s" % (self.step, self.feedback)
                _id = 0
                for bot in self.bots:
                    bot.controller._feedback(False, self.feedback, _id) #if shared controller this SHOULDN'T BREAK ANYTHING
                    _id += 1
        self.step += 1

        collision = {}
        ind = 0
        for bot in self.bots:
            if bot.controller:
                sensors = self.sensor_func(self, bot)
                sensors.append(1.0)
                instructions = bot.controller(sensors, ind)
                if instructions is not None:
                    print "INSTRUCTIONS: , ", [i for i in instructions]
                #dist, color = bot.rangefinder(self) #gets distance and color from rangefinder, now
                #instructions = bot.controller([dist, color[0], color[1], color[2], 1.0]) #TODO: Add sensory inputs here
                    collision = self.movement_func(self, bot, instructions, collision)
            ind += 1
        if self.player:
            for player in self.player:
                collision = self.movement_func(self, player, self.player_instructions, collision)
        #for c in range(len(collision)):
        #    net_x = sum([i[0] for i in collision.values()[c]])
        #    net_y = sum([i[1] for i in collision.values()[c]])
        #    #print "NET_X: %x   NET_Y : %x" % (net_x, net_y)
        #    collision.keys()[c].move(self, net_x, net_y)
        while len(collision) is not 0:
            c = collision.keys()[0]
            net_x = sum([i[0] for i in collision.values()[0]])
            net_y = sum([i[1] for i in collision.values()[0]])
            #print "NET_X: %x   NET_Y : %x" % (net_x, net_y)
            collision.pop(c)
            collision = c.move(self, net_x, net_y, collision)
            

            


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

def generate_positions_by_minimum_distance_with_obstacles(environment, shape, num_bots, min_distance):
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
        for c in environment.collidables:
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

def generate_positions_within_region_with_obstacles(environment, shape, num_bots, region):
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
        for c in environment.collidables:
            if c.rect.colliderect(pygame.Rect(x, y, 16, 16)):
                collide = True
        #
        dist = None
        if len(x_pos) > 0:
            dist = max([sqrt((x-i)**2 + (y - j)**2) for i, j in zip(x_pos, y_pos)])
        if region.collidepoint(x, y) and not collide: #hardcode distance min
            if dist == None or dist > 30:
                x_pos.append(x)
                y_pos.append(y)

    if len(x_pos) < num_bots:
        print "SHIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIi", zip(x_pos, y_pos)
        return None
    else:
        #print "Remaining points: %s %s" % (len(x_points), len(y_points))
        return zip(x_pos, y_pos)



def initialize_collidable_obstacles(env, ENV_SHAPE, num, mean_width, w_var, mean_height, h_var, min_space = 100):
    collidables = []
    pos = None
    while pos is None:
        pos = generate_positions_by_minimum_distance_with_obstacles(env, ENV_SHAPE, num, min_space)
    for x, y in pos:
        height = mean_height + h_var * random.random() * random.choice((1, -1))
        width = mean_width + w_var * random.random() * random.choice((1, -1))
        collidables.append(Collidable(x, y, height, width, white))
    return collidables



def move_bots(env, bot, instructions = [], collision = []):
    hit_wall = False
    res = bot.move(instructions[0], instructions[1], instructions[2])
    tmp_x = bot.x + res[0]
    tmp_y = bot.y + res[1]
    try:
        tmp_rect = pygame.Rect(tmp_x, tmp_y, bot.rect_h, bot.rect_w)
        for c in env.collidables:
            if tmp_rect.colliderect(c.rect) and c != bot:
                if hasattr(c, 'moveable') and c.moveable == True: #collided with moveable object
                    collision.setdefault(c, []).append((res[0], res[1]))
                else: hit_wall = True
        if not len(collision) and not hit_wall:
            bot.x = tmp_x
            bot.y = tmp_y
        bot.rect = pygame.Rect(bot.x, bot.y, bot.rect_h, bot.rect_w)
    except: #getting weird error with the tmp_rect initialization
        pass
    return collision


def get_distance_between_rect(a, b):
    if a.colliderect(b):
        return 0.0
    else:
        return sqrt((a.x - b.x) **2 + (a.y - b.y) **2)

def feedback_by_min_relative_bot_distances(environment, *args):
    feedback = 0
    for i in range(len(environment.bots)): #for now we just space them horizontally 
        closest = float('inf')
        for j in range(len(environment.bots)): #TODO NOW WE MIMIZE DISTANCE TO CLOSEST BOT
            if j != i:
                distance = sqrt((environment.bots[i].x - environment.bots[j].x) ** 2 + 
                        (environment.bots[i].y - environment.bots[j].y) ** 2)
                if distance < closest:
                    closest = distance
        feedback += closest #should never be inf
    return feedback

def feedback_by_sum_relative_box_distances(environment, *args):
    dist = 0
    for i in range(len(self.bots)):
        for j in range(len(self.bots)): #TODO NOW WE MIMIZE DISTANCE TO CLOSEST BOT
            if j != i:
                dist += get_distance_between_rect(self.bots[i].rect, self.bots[j].rect) 
    return dist


def feedback_by_moveable_distance_to_region(environment, region): #region is a Rect 
    dist = 0
    for c in environment.collidables:
        if hasattr(c, 'moveable') and c.moveable:
            dist += get_distance_between_rect(c.rect, region)
    return dist

def feedback_by_total_bot_distance_from_region(environment, region):  #region is a Rect
    dist = 0
    for b in environment.bots:
        dist += get_distance_between_rect(b.rect, region)
    return dist



def reset_after_clock_threshold(env, threshold = 2000):
    return env.step >= threshold    

def rangefinder_sensor(env, bot, *args):
    dist, color = bot.rangefinder(env)
    dist = [dist,]
    dist.extend(color)
    return dist

def rangefinder_orientation_sensor(env, bot, *args):
    dist = rangefinder_sensor(env, bot, *args)
    dist.append(bot.orientation)
    return dist

def prickle_sensor(env, bot, *args):
    dist, color = bot.rangefinder(env)
    sense = []
    for i in range(len(dist)):
        sense.append(dist[i])
        sense.extend(color[i])
    #print "SENSE: ", sense
    return sense

   
def convolutional_sensor(env, bot, *args):
    rect = pygame.Rect(25, 25, 100, 50)
    screenshot = pygame.Surface(100, 50)
    screenshot.blit(screen, area=rect)
    pygame.image.save(screenshot, "screenshot.jpg")

if __name__ == '__main__':
    #SCREENSIZE = [1360, 700]
    SCREENSIZE = [700, 700]
    COLLISION_PENALTY = 0.1 #penalize collisions, straight-up
    STATIONARY_PENALTY = 1 #penalize non-moving bots
    STATIONARY_THRESHOLD = 20 #time-threshold for immobile bots
    ROTATE_THRESHOLD = 4 #RADIANS
    ROTATE_PENALTY = 1

    INVALID_REVERSE_PENALTY = 2 #penalize moving in reverse without object within "threshold" of rangefinder
    INVALID_REVERSE_THRESHOLD = 15 #allowable distance at which to "allow" reverse

    DURATION = 600 
    SPEED = 1000

    MINIMUM_BOT_DISTANCE = 200 #used for initializing the bots with random-but-spaced points

    bots = []
    NUM_BOTS = 2
    #controller = DumbController()
    try: controller = MultiNEATController(NEAT_WRAPPER, NUM_BOTS)
    except: controller = DumbController()

    for i in range(NUM_BOTS): #for now we just space them horizontally 

        bots.append(LinearPrickleBot(7, 100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
    env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_total_bot_distance_from_region,# 
            movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = prickle_sensor,
            default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    #env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_moveable_distance_to_region,# 
    #        movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = prickle_sensor,
    #        default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    #controller = MultiNEATController(NEAT_WRAPPER, NUM_BOTS)

    #for i in range(NUM_BOTS): #for now we just space them horizontally 

    #    #bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controllers[i]))
    #    bots.append(LinearRangefinderBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
    #                    #TODO: Fix hitboxes on ALL objects to their visuals correspond to their actual hitboxes (instead of ... not)
    #    #bots.append(LinearPrickleBot(4, 100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
    #env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_total_bot_distance_from_region,# 
    #        movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = rangefinder_sensor,
    #        default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    #env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_moveable_distance_to_region,# 
    #        movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = rangefinder_orientation_sensor,
    #        default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    env.play() 



    
