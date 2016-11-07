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
	c = 0.0005 # iteration rate
	P = 800 * np.random.rand(D, N) # Initial positions of robots in a 10x10 unit space
	Pn = np.zeros((D,N), dtype=np.int)# Position bucket I THINK...

	P0 = 800 * np.random.rand(D,1)
	P0n = np.zeros((D,1), dtype=np.int)
	Y = [500,500]

	error = float("inf") # Initial error is HUGE. We want to minimize this.
	goal = 0.001# The final error should be really small
	steps = 0 # TIME STEPS cuz computer reasons.

	SCREENSIZE = [800, 800] # Size of our output display
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
				P0n = P0
				steps += 1

				P0n = P0n - c* (P0 -Y)
				pygame.draw.circle(screen, black, [int(P0[0, 0]), int(P0[1, 0])], 7, 1)

				for i in range(N):
					for j in range(N):
						pygame.draw.circle(screen, black, [int(P[0, i]), int(P[1, i])], 7, 1)
						#pygame.draw.circle(screen, black, [int(P0[0, i]), int(P0[1, i])], 7, 1)

						Pn[:, i] = Pn[:, i] + c * (P[:, j]-P[:, i])  - c * (P[:,j] - P0[:,0])

						pygame.draw.circle(screen, green, [int(P[0,i]),int(P[1,i])] ,7, 1)
						#pygame.draw.circle(screen, red, [int(P0[0, 0]), int(P0[1, 0])], 7, 1)


				P = Pn
				P0 = P0n
				pygame.draw.circle(screen, red, [int(P0[0, 0]), int(P0[1, 0])], 7, 1)
				n_error = np.linalg.norm(P0n -Y)

				error = n_error
				print ('ERROR: ', error)
				pygame.display.update()


			print ("Consensus Reached in ", steps, "steps")
			input("Ctrl+C to exit")

	except KeyboardInterrupt:
		print (" Shutting down simulation...")
		pygame.quit()
