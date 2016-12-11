import pygame
from pygame.locals import *
import random

import numpy as np
from math import sin, cos, pi, floor, sqrt, floor
from abc import abstractmethod


from sim import Controller, Environment, DumbController, LinearRangefinderBot, Collidable, Moveable , LinearPrickleBot

red = [255,0,0]
green = [0,255,0]
blue = [0,0,255]
white = [255,255,255]
black = [0,0,0]

## KERAS WRAPPER AND CONTROLLER HERE

import keras
from keras.layers import Dense, Reshape, Merge, Activation, Flatten
from keras.optimizers import SGD
model = keras.models.Sequential()

model.add(Dense(15, input_dim = 32, init = 'uniform', activation = 'tanh'))
model.add(Dense(20, init = 'uniform', activation = 'sigmoid'))
#model.add(keras.layers.recurrent.SimpleRNN(3))    
#model.add(Activation('tanh'))
model.add(Dense(1, init = 'uniform', activation = 'linear'))

sgd = SGD(lr = 0.01, decay = 0.00001, momentum = 0.9)
model.compile(optimizer=sgd, loss='mse')
print model.summary()
print model.inputs
print model.outputs

raw_input("Press Enter to continue...")

class KerasDQNController(Controller): 
    def __init__(self, action_list, model, policy = 'eps', eps = 0.2, eps_decay = 0.01, gamma = 0.8): 
        self.model = model
        self.policy = policy
        self.action_list = action_list
        self.eps = eps #initial (or static) epsilon value for exploration
        self.eps_decay = eps_decay
        self.gamma = gamma #discount value
        self.steps = 0

    def __call__(self, senses, bot_id):
        if self.policy == 'eps' or self.policy == 'anneal':
            if random.random() >= self.eps: #SELECT GREEDY ACTION (greed) 
                print "SENSES: ", len(senses)
                action = self._get_minimizing_action(senses)
                #print "ACTION: ", action
                return action
            else: #SELECT RANDOM ACTION (explore)
                return np.array([random.choice((-1, 1)) for i in range(3)])


    def _feedback(self, terminal = False, *args):
        pass
    
    def _train(self):
        pass

    def _calculate_expected_q(self, reward, next_state):
        pass

    def _get_minimizing_action(self, state):
        minimizing = None
        minimum = float('inf') 
        for i in self.action_list[0]:
            for j in self.action_list[1]:
                for k in self.action_list[2]:
                    tmp = state[:]
                    tmp.extend([i, j, k])
                    tmp = np.array(tmp).reshape(1, 29 + 3)
                    #print "TEMP INPUT: ", tmp
                    #print "SHAPE: ", tmp.shape
                    val = self.model.predict([tmp,])
                    #print "VAL: ", val
                    if val[0][0] < minimum:
                        minimizing = [i, j, k]
                        minimum = val[0][0]
        return minimizing

    def _get_q_value(self, state, action):
        tmp = state[:]
        tmp.extend(action)
        tmp = np.array(tmp).reshape(1, 29 + 3)
        return self.model.predict([tmp,])







class KerasDQNERController(KerasDQNController):
    def __init__(self, num_samples, limit, overwrite_prob, *args, **kwargs):
        KerasDQNController.__init__(self, *args, **kwargs)
        self.memory_limit = limit
        self.memory = [] #structure: state_t, action_t, reward_t, state_t+1, terminal_boolean
        self.overwrite_prob = overwrite_prob
        self.num_samples = num_samples

        self.state_queue = []
        self.action_queue = []
        self.feedback_queue = {}

    def __call__(self, senses, bot_id):
        self.steps += 1
        #senses = senses[:len(senses) - 1] #disregard biasing input(?) <3
        self.state_queue.append(senses)
        action = KerasDQNController.__call__(self, senses, bot_id) 
        #action = np.array([random.choice((-1, 1)) for i in range(3)]) #UNTIL Q-Function retrieval works
        self.action_queue.append(action)
        return action


    def _feedback(self, terminal = False, *args):
        print "FOR BOT: ", args[1]
        try: 
            self.feedback_queue[args[1]]
            if args[0] <= 0:  #minimal value is 0 for these tasks
                pass
            elif self.steps > 1: #for now, we disregard the 1st state transition entirely
                try:
                    _id = args[1] 
                    feedback = args[0] - self.feedback_queue[args[1]]
                    print "Q-Feedback: ", feedback
                    print "fb: %s queue: %s" % (feedback, self.feedback_queue)
                    #print "FEEDBACK: %s - %s = %s" % (args[0], self.prev_feedback, feedback)
                    if terminal: #add another memory     
                        exp = (self.state_queue.pop(0), self.action_queue.pop(0), feedback, None, terminal) 
                        self.memory.append(exp)
                        self._train() #CALL TRAIN HERE, THIS SHOULD CLEAR THE MEMORY FROM THE PREVIOUS EPOCH <3
                    else:
                        exp = (self.state_queue.pop(0), self.action_queue.pop(0), feedback, self.state_queue[0], terminal) 
                        if len(self.memory) <= self.memory_limit: 
                            self.memory.append(exp)
                        elif random.random() < self.overwrite_prob: #randomly assign
                            ind = random.choice(range(len(self.memory) - 1))
                            self.memory.pop(ind)
                            self.memory.insert(ind, exp)
                except IndexError:
                    print "INDEXERROR"
            self.feedback_queue[args[1]] = args[0]
        except KeyError: 
            self.feedback_queue[args[1]] = args[0]
        

    def _train(self):
        print "TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        #print self.memory[len(self.memory) - 1] #TODO; train with minibatches across all sampled inputs instead of one-at-a-time
        num_samples = self.num_samples
        if self.num_samples > len(self.memory):
            num_samples = len(self.memory)
        for i in range(num_samples):
            if len(self.memory) <= 1:
                break
            ind = random.choice(range(len(self.memory) - 1))
            exp = self.memory.pop(ind)
            y = None #expected reward
            if exp[4] == True: #terminal state
                y = exp[2] #set to the reward of the terminal state
            else: #non-terminal state
                #print "STATE: ", exp[3]
                y = exp[2] + self.gamma * self._get_q_value(exp[3], self._get_minimizing_action(exp[3])) # y = r_t + maximum Q(max_a, s_t+1) 
            inp = exp[0][:]
            inp.extend(exp[1])
            inp = np.array(inp).reshape(1, 29 + 3)
            #print "INPUT: ", inp
            try: self.model.fit(np.array(inp), np.array(y), nb_epoch = 1)
            except: pass
        

        if self.policy == 'anneal' and self.eps > 0:
            self.eps = max(0, self.eps - self.eps_decay) #clamp it at 0
        self.memory = [] #clear memory 
        #raw_input("TRAINED! Press Enter to continue...")


## END KERAS



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

def initialize_collidable_obstacles(env, ENV_SHAPE, num, mean_width, w_var, mean_height, h_var, min_space = 100):
    collidables = []
    pos = generate_positions_by_minimum_distance_with_obstacles(env, ENV_SHAPE, num, min_space)
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
    for c in env.collidables:
        if tmp_rect.colliderect(c.rect) and c != bot:
            if hasattr(c, 'moveable') and c.moveable == True: #collided with moveable object
                collision.setdefault(c, []).append((res[0], res[1]))
            else: hit_wall = True
    if not len(collision) and not hit_wall:
        bot.x = tmp_x
        bot.y = tmp_y
    bot.rect = pygame.Rect(bot.x, bot.y, bot.rect_h, bot.rect_w)
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

def prickle_sensor(env, bot, *args):
    dist, color = bot.rangefinder(env)
    sense = []
    for i in range(len(dist)):
        sense.append(dist[i])
        sense.extend(color[i])
    return sense
    

if __name__ == '__main__':
    SCREENSIZE = [700, 700]
    COLLISION_PENALTY = 0.1 #penalize collisions, straight-up
    STATIONARY_PENALTY = 1 #penalize non-moving bots
    STATIONARY_THRESHOLD = 20 #time-threshold for immobile bots
    ROTATE_THRESHOLD = 4 #RADIANS
    ROTATE_PENALTY = 1

    INVALID_REVERSE_PENALTY = 2 #penalize moving in reverse without object within "threshold" of rangefinder
    INVALID_REVERSE_THRESHOLD = 15 #allowable distance at which to "allow" reverse

    DURATION = 1000 
    SPEED = 1000

    MINIMUM_BOT_DISTANCE = 300 #used for initializing the bots with random-but-spaced points

    bots = []
    NUM_BOTS = 2
    controller = DumbController()
    #controller = MultiNEATController(NEAT_WRAPPER, NUM_BOTS)
    controller = KerasDQNERController(int(DURATION / 4), float('inf'), 0.2, [[-1, 1] for i in range(3)], model, policy = 'anneal', eps = .7, 
            eps_decay = 0.0001)
    #for i in range(NUM_BOTS): #for now we just space them horizontally 

    #    #bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controllers[i]))
    #    bots.append(LinearRangefinderBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
    #                    #TODO: Fix hitboxes on ALL objects to their visuals correspond to their actual hitboxes (instead of ... not)
    #env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_moveable_distance_to_region,#feedback_by_total_bot_distance_from_region,# 
    #        movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = rangefinder_sensor,
    #        default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    #env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_total_bot_distance_from_region,# 
    #        movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = rangefinder_sensor,
    #        default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    #env.play() 
    for i in range(NUM_BOTS): #for now we just space them horizontally 

        #bots.append(TankDriveBot(100 * i + 50, 50 * i + 50, 3, 0.2, 0, controllers[i]))
        bots.append(LinearPrickleBot(7, 100 * i + 50, 50 * i + 50, 3, 0.2, 0, controller))
                        #TODO: Fix hitboxes on ALL objects to their visuals correspond to their actual hitboxes (instead of ... not)
    
    #env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_moveable_distance_to_region,#feedback_by_total_bot_distance_from_region,# 
    #        movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = prickle_sensor,
    #        default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    env = Environment(SCREENSIZE, SPEED, controller, bots, feedback_func = feedback_by_total_bot_distance_from_region,# 
            movement_func = move_bots, reset_func = reset_after_clock_threshold, sensor_func = prickle_sensor,
            default_clock_threshold = DURATION, default_region = pygame.Rect(0, 0, 220, 220))
    env.play() 



    
