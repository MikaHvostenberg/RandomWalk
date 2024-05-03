import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt


def update_values(ballveloc:np.array, scaling:float):
    """
    Updates the x,y,z values of the particle with the exp(-x) probability distribution 
    when it hits the box floor (z==0).
    """

    plusminus = np.random.randint(0,2,2)*2-1
    plusminus = np.append(plusminus, 1)
    newvi = ballveloc + plusminus*sp.stats.expon.rvs(loc=0, scale=scaling, size=3, random_state=None)

    return newvi



if __name__ == "__main__":
    downacc = 0.01
    ttotal = 10
    tstep = 0.1
    i = 0
    pos = np.array([0,0,0], dtype=float)
    vel = np.array([0,1,-1], dtype=float)
    
    out_xlist = np.array([])
    out_ylist = np.array([])
    out_zlist = np.array([])



    while i <= ttotal:
        vel[2] += -downacc*tstep
        pos += vel*tstep

        if pos[2] <= 0:
            pos[2] = 0
            vel = update_values(vel, 0.1)

        out_xlist = np.append(out_xlist, pos[0])
        out_ylist = np.append(out_ylist, pos[1])
        out_zlist = np.append(out_zlist, pos[2])

        i += tstep
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(out_xlist, out_ylist, out_zlist)
    fig.savefig("scatterplot.pdf")

    exit()