import math
import pygame


N = 12 # Number of Robots
M = 2  # Dimension of search space
c = 0.01 #itteration rate
P = 10*rand(M,N) 
Pn = z

error = float("inf")
goal = 0.1 
steps = 0

A=round(rand(N,N));
A=round((A+A')/2);
A=max(eye(N),A);



figure(1);
plot(P(1,:),P(2,:),'ob');
axis([0 10 0 10]);
title('Randomized Initial Particle Positions in a 2D Space')
while error>goal:
	Pn=P; steps=steps+1
	for i=1:N
		for j=1:N
			Pn(:,i)=Pn(:,i)+c*((P(:,j)-P(:,i))*A(i,j))


P=Pn;
figure(2);
plot(P(1,:),P(2,:),'ob');
axis([0 10 0 10]);

title('Particles Reach Consenus')
F(steps) = getframe;
n_error=norm(var((Pn-repmat(mean(Pn')',[1 N]))'));
error=n_error;

fprintf('Error: %f\n',error)
end
fprintf('Consensus reached in %i steps.',steps);
movie2avi(F,'Consensus Equ