import matplotlib.pyplot as plt
from distributions.weibull import *
from distributions.lomax import *
from distributions.lognormal import *
from distributions.loglogistic import *
from nonparametric.non_parametric import *
from markovchains.markovchains import *
from datetime import datetime
from numpy import genfromtxt
import os
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


k=1.1; lmb=0.5; intervention_cost=200; sample_size=5000; censor_level=200; prob=1.0
l = Lomax(k=k, lmb=lmb)
samples = l.samples(size=sample_size)
unifs = np.random.uniform(size=sample_size)
ti = samples[(samples<=censor_level) + (unifs>prob)]
xi = np.ones(sum( (samples>censor_level) * (unifs<=prob)))*censor_level
ll1 = LogLogistic(ti=ti, xi=xi)

alphas = np.arange(0.01,50.0,0.1)
betas = np.arange(0.01,3.0,2.99/(len(alphas)-0.5))
X = np.array([alphas,]*len(alphas)).T
Y = np.array([betas,]*len(alphas))
#Z = np.zeros(X.shape)
Z1 = np.zeros(X.shape)
params = np.array([ 27.2716913,   0.14728398])
H = ll1.hessian(ti,xi,params[0],params[1])
g = ll1.grad(ti,xi,params[0],params[1])
f = ll1.loglik(ti,xi,params[0],params[1])
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		#Z[i,j] = ll1.loglik(ti,xi,X[i,j],Y[i,j])
		dely = Y[i,j]-params[1]
		delx = X[i,j]-params[0]
		Z1[i,j] = (H[0,0]*delx**2+H[1,1]*dely**2+2*H[0,1]*delx*dely)/2+g[0]*delx+g[1]*dely+f
(x,y,z) = (ll1.alpha,ll1.beta,ll1.loglik(ti,xi,ll1.alpha,ll1.beta))
(x1,y1,z1) = (params[0],params[1],ll1.loglik(ti,xi,params[0],params[1]))
plot_fig(X,Y,Z,Z1,(x,y,z),(x1,y1,z1))



def plot_fig(X,Y,Z,Z1,pt1=None,pt2=None):
	'''
	See some sample test data:
	X, Y, Z = axes3d.get_test_data(0.05)
	'''
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# Plot a basic wireframe.
	ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
	#cset = ax.contour(X, Y, Z1, cmap=cm.coolwarm)
	#ax.clabel(cset, fontsize=9, inline=1)
	#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
	#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
	#cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
	#cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
	if pt1 is not None:
		ax.scatter([pt1[0]], [pt1[1]], [pt1[2]], color='r')
	if pt2 is not None:
		ax.scatter([pt2[0]], [pt2[1]], [pt2[2]], color='g')
	plt.show()


def plot_quad():
	alphas = np.arange(1.0,200.0,3.0)
	betas = np.arange(0.01,5.0,4.99/(len(alphas)-0.5) )
	X = np.array([alphas,]*len(alphas)).T
	Y = np.array([betas,]*len(alphas))
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			dely = Y[i,j]-params[1]
			delx = X[i,j]-params[0]
			Z1[i,j] = (H[0,0]*delx**2+H[1,1]*dely**2+2*H[0,1]*delx*dely)/2+g[0]*delx+g[1]*dely+f

def quad(x,y):
	delx = x-params[0]
	dely = y-params[1]
	return (H[0,0]*delx**2+H[1,1]*dely**2+2*H[0,1]*delx*dely)/2+g[0]*delx+g[1]*dely



alphas = np.arange(-200,200.0,5.0)
betas = np.arange(-200,200.0,5.0)
X = np.array([alphas,]*len(alphas)).T
Y = np.array([betas,]*len(alphas))
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
	for j in range(X.shape[1]):
		Z[i,j] = -X[i,j]**2-Y[i,j]**2
plot_fig(X,Y,Z,Z,(0,0,0))


