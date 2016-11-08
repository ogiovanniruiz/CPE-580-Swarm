import pygame
from pygame.locals import *

import numpy as np

red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
white = [255, 255, 255]
black = [0, 0, 0]

SCREENSIZE = [800, 800]  # Size of our output display
screenBGColor = black  # White background

N = 20  # Number of Robots
D = 2  # Dimension of search space
c = 0.0005  # iteration rate
c0 = 0.005 # leader iteration rate
Y = [500, 500] # final position of leader
goal = 1  # The final error should be really small
running = True

class Swarm_Simulation:
	def __init__(self):

		pygame.init()  # Shall we begin?
		self.screen = pygame.display.set_mode(SCREENSIZE)
		self.screen.fill(screenBGColor)
		pygame.display.set_caption("Rendezvous Simulation")

		self.error = float("inf")  # Initial error is HUGE. We want to minimize this.
		self.steps = 0

		self.P = 800 * np.random.rand(D, N)  # Initial positions of robots in a 10x10 unit space
		self.Pn = np.zeros((D, N), dtype=np.int)  # Position bucket I THINK...


		self.P0 = 800 * np.random.rand(D, 1)
		self.P0n = np.zeros((D, 1), dtype=np.int)


	def Run(self):



		while (running):

			while self.error > goal:
				self.Pn = self.P
				self.P0n = self.P0
				self.steps += 1

				self.P0n = self.P0n - c0 * (self.P0 - Y)
				pygame.draw.circle(self.screen, black, [int(self.P0[0, 0]), int(self.P0[1, 0])], 7, 1)

				for i in range(N):
					for j in range(N):
						pygame.draw.circle(self.screen, black, [int(self.P[0, i]), int(self.P[1, i])], 7, 1)

						self.Pn[:, i] = self.Pn[:, i] + c * 0.1 * (self.P[:, j] - self.P[:, i]) - c * (self.P[:, j] - self.P0[:, 0])

						pygame.draw.circle(self.screen, green, [int(self.P[0, i]), int(self.P[1, i])], 7, 1)

				self.P = self.Pn
				self.P0 = self.P0n
				pygame.draw.circle(self.screen, red, [int(self.P0[0, 0]), int(self.P0[1, 0])], 7, 1)

				n_error = np.linalg.norm(self.P0n - Y)

				self.error = n_error
				print('ERROR: ', self.error)
				pygame.display.update()

			print("Consensus Reached in ", self.steps, "steps")
			input("Ctrl+C to exit")


if __name__ == '__main__':

	try:
		sim	= Swarm_Simulation()
		sim.Run()

	except KeyboardInterrupt:
		print (" Shutting down simulation...")
		pygame.quit()
