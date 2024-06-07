from os import ttyname
import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
from datafit import compute_qk


def update_values(ballveloc:np.ndarray, scaling:float):
    """
    Updates the x,y,z values of the particle with the exp(-x) probability distribution 
    when it hits the box floor (z==0).
    """

    plusminus = np.random.randint(0,2,2)*2-1
    plusminus = np.append(plusminus, 1)
    newvi = ballveloc + plusminus*sp.stats.expon.rvs(loc=0, scale=scaling, size=3, random_state=None)

    return newvi


def make_simulation() -> tuple[np.ndarray, np.ndarray]:
    downacc = 0.01
    ttotal = 1000
    tstep = 0.2
    floorheight = 0.0025
    intensity= 0.1
    damping= 0.5
    boxlength = 5
    boxlengtherror = 0.1
    boxwidth = 5
    wallheight = 3

    n = 300
    velfactor = 10 # how much to shrink the initial normal dist values by
    pos = np.zeros((n,3), dtype=float)
    vel = np.random.randn(n,3)/velfactor
    
    # out_xlist = np.array([])
    # out_ylist = np.array([])
    # out_zlist = np.array([])

    i = 0
    counterlist = []
    print("Beginning simulation.")
    while i <= ttotal:

        vel[:,2] += -downacc*tstep
        # print(vel)
        # exit()

        pos += vel*tstep
        # print(pos)
        # exit()

        counter0 = 0
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        counter5 = 0
        counter6 = 0
        for j, partipos in enumerate(pos):

            if partipos[2] <= floorheight:
                pos[j,2] = 0
                vel[j] = update_values(damping*np.abs(vel[j]), intensity)
            if abs(partipos[0]) >= boxwidth:
                vel[j][0] *= -1
            if partipos[1] < 0:
                pos[j,1]   = 0
                vel[j][1] *= -1
            if partipos[1]%boxlength <= boxlengtherror and partipos[2] < wallheight:
                vel[j][1] *= -1

            match partipos[1]//boxlength:
                case 0: counter0 += 1
                case 1: counter1 += 1
                case 2: counter2 += 1
                case 3: counter3 += 1
                case 4: counter4 += 1
                case 5: pos[j] = [0.0,0.0,0.0]; counter0 += 1
                case _: pos[j] = [0.0,0.0,0.0]; counter0 += 1
        
        # steady_state
        if counter0 < n:
            pos = np.append(pos,[[0.0,0.0,0.0]],axis=0)
            vel = np.append(vel,np.random.randn(1,3)/velfactor,axis=0)


        counterlist.append([counter0,counter1,counter2,counter3,counter4,counter5,counter6])

        # out_xlist = np.append(out_xlist, pos[:,0])
        # out_ylist = np.append(out_ylist, pos[:,1])
        # out_zlist = np.append(out_zlist, pos[:,2])

        if int(1/tstep*round(i,1))%int(10/tstep**2)==0:
            print(f"Step \t{int(round(i,0))}/{ttotal} completed.")

            print(f"Particles: \tn[0|1|2|>2]=[{counter0} \t{counter1} \t{counter2} \t{counter3} \t{counter4} \t{counter5} \t{counter6}]")

        i += tstep
     
    print("Simulation completed.")
    return np.array(counterlist), np.linspace(0,ttotal,int(ttotal/tstep))




if __name__ == "__main__":
    
    avgnum = 5
    counterarray, tvals = make_simulation()
    for inst in range(avgnum-1):
        counterarray += make_simulation()[0]
    counterarray = counterarray/avgnum/300
    
    colors = ['b','g','r','c','m','y']
    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,constrained_layout=True)
    # fig.suptitle(f"Physical simulation of {n}\nparticles in infinitely many boxes")

    ax.plot(tvals,counterarray[:,0], label="$1$", c=colors[0])
    ax.plot(tvals,counterarray[:,1], label="$2$", c=colors[1])
    ax.plot(tvals,counterarray[:,2], label="$3$", c=colors[2])
    ax.plot(tvals,counterarray[:,3], label="$4$", c=colors[3])
    ax.plot(tvals,counterarray[:,4], label="$5$", c=colors[4])
    ax.plot(tvals,counterarray[:,5], label="$6$", c=colors[5])
    # ax.plot(tvals,counterarray[:,6], label="Boxes $>6$")

    ax.legend(loc="upper right", fontsize="small", title="Bin $k$")    

    fig.supxlabel("Time")
    fig.supylabel("Normalised Nr. of particles")

    for k in range(2,6):
        paramd, pcovd = sp.optimize.curve_fit(lambda t,D: compute_qk(t,D,k),tvals,counterarray[:,k-1], bounds=[0,0.0001])
        perrod = np.sqrt(np.diag(pcovd))
        print(f"For k={k}, D = {paramd} +- {perrod}")

        ax.plot(tvals, compute_qk(tvals,paramd,k),c=colors[k-1],linestyle="dotted")
    
    fig.savefig("plots/ballsinbins-steady-avg.pdf")
    np.savetxt("datacounter.csv",counterarray,delimiter=',')
    exit()
