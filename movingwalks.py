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
    ttotal = 300
    tstep = 0.1
    floorheight = 0.0025
    intensity= 0.1
    damping= 0.5
    boxlength = 5
    boxlengtherror = 0.1
    boxwidth = 5
    wallheight = 3

    n = 100
    velfactor = 10 # how much to shrink the initial normal dist values by
    pos = np.zeros((n,3), dtype=float)
    vel = np.random.randn(n,3)/velfactor
    
    out_xlist = np.array([])
    out_ylist = np.array([])
    out_zlist = np.array([])

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

        counterlist.append([counter0,counter1,counter2,counter3])

        out_xlist = np.append(out_xlist, pos[:,0])
        out_ylist = np.append(out_ylist, pos[:,1])
        out_zlist = np.append(out_zlist, pos[:,2])

        if int(1/tstep*round(i,1))%int(1/tstep**2)==0:
            print(f"Step \t{int(round(i,0))}/{ttotal} completed.")

            print(f"Particles: \tn[0|1|2|>2]=[{counter0} \t{counter1} \t{counter2} \t{counter3}]")

        i += tstep
     
    print("Simulation completed.")
    counterarray = np.array(counterlist)

    fig, ax = plt.subplots(1,1,figsize=(4,4),sharex=True,sharey=True)
    fig.suptitle(f"Physical simulation of {n}\nparticles in infinitely many boxes")

    ax.plot(counterarray[:,0], label="Box \t$0$")
    ax.plot(counterarray[:,1], label="Box \t$1$")
    ax.plot(counterarray[:,2], label="Box \t$2$")
    ax.plot(counterarray[:,3], label="Boxes $>2$")

    ax.legend(loc="upper right", fontsize="small")    

    fig.supxlabel("Timestep")
    fig.supylabel("Nr. of particles")

    fig.savefig("ballsinbins.pdf")
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(out_xlist, out_ylist, out_zlist)

    # fig.savefig("scatterplot.pdf")
    # plt.show()

    exit()