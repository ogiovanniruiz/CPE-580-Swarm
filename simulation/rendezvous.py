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
	N = 12  # Number of Robots
	D = 2  # Dimension of search space
	c = 0.01  # iteration rate
	P = 800 * np.random.rand(D, N) # Initial positions of robots in a 10x10 unit space
	Pn = np.zeros((D,N), dtype=np.int)# Position bucket I THINK...

	error = float("inf") # Initial error is HUGE. We want to minimize this.
	goal = 100# The final error should be really small
	steps = 0 # TIME STEPS cuz computer reasons.

	SCREENSIZE = [800, 600] # Size of our output display
	screenBGColor = white # White background
	running = True

	x = 0
	y = 0
	pygame.init() # Shall we begin?

	try:
		while(running):
			screen = pygame.display.set_mode(SCREENSIZE)
			screen.fill(screenBGColor)
			pygame.display.set_caption("Rendezvous Simulation")

			while error > goal:
				Pn = P
				steps += 1

				for i in range(N):
					for j in range(N):
						pygame.draw.circle(screen, blue, [x, y], 7, 1)
						x = int(P[0, i])
						y = int(P[1, i])

						Pn[:, i] = Pn[:, i] + c * (P[:, j]-P[:, i])

						pygame.display.update()

				screen.fill(screenBGColor)
				P = Pn
				n_error = np.linalg.norm(scipy.spatial.distance.cdist(Pn,Pn))

				error = n_error
				print ('ERROR: ', error)


			print ("Consensus Reached")
			input("Ctrl+C to exit")

	except KeyboardInterrupt:
		print (" Shutting down simulation...")
		pygame.quit()
