import pygame
from pygame.locals import *

import numpy as np
import scipy.spatial.distance


red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
white = [255, 255, 255]
black = [0, 0, 0]

if __name__ == '__main__':
	N = 20# Number of Robots
	D = 2  # Dimension of search space
	c = 0.001  # iteration rate
	P = 800 * np.random.rand(D, N) # Initial positions of robots in a 10x10 unit space
	Pn = np.zeros((D,N), dtype=np.int)# Position bucket I THINK...

	error = float("inf") # Initial error is HUGE. We want to minimize this.
	goal = 100# The final error should be really small
	steps = 0 # TIME STEPS cuz computer reasons.

	SCREENSIZE = [800, 600] # Size of our output display
	screenBGColor = black # White background
	running = True

	pygame.init() # Shall we begin?
	screen = pygame.display.set_mode(SCREENSIZE)
	screen.fill(screenBGColor)
	pygame.display.set_caption("Rendezvous Simulation")

	try:
		while(running):

			while error > goal:
				Pn = P
				steps += 1

				for i in range(N):
					for j in range(N):
						circle = pygame.draw.circle(screen, black, [int(P[0, i]), int(P[1, i])], 7, 1)

						Pn[:, i] = Pn[:, i] + c * (P[:, j]-P[:, i])

						circle = pygame.draw.circle(screen, green, [int(P[0,i]),int(P[1,i])] ,7, 1)


				P = Pn
				n_error = np.linalg.norm(scipy.spatial.distance.cdist(Pn,P))

				error = n_error
				print ('ERROR: ', error)
				pygame.display.update()


			print ("Consensus Reached")
			input("Ctrl+C to exit")

	except KeyboardInterrupt:
		print (" Shutting down simulation...")
		pygame.quit()
