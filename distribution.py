import numpy as np
import matplotlib.pyplot as plt
import math


# computation of Q_k(t)


if __name__ == "__main__":
    pi = math.pi
    nbins = 6 # number of bins
    param_D = 0.005

    numterms = 3000
    # timetotal = np.concatenate(
    #     (
    #         np.linspace(0,1,40),
    #         np.linspace(1,5,80),
    #         np.linspace(5,10,40),
    #         np.linspace(10,20,40),
    #         np.linspace(20,50,80),
    #         np.linspace(50,100,40),
    #     ),
    #     axis=None
    # )
    timetotal = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,24,48,96,192,384])

    # iterate over each of the bins and then for each time value compute the series
    qvals = np.zeros((len(timetotal), nbins))
    for i,k in enumerate(range(1,nbins+1)):
        # bin index i and k in formula
        qvals[:,i] += (13-2*k)/11

        for j,t in enumerate(timetotal):
            term = 0
            for n in range(1,numterms+1):
                term += 1/(n**2*pi**2) * (1+11*math.cos(n*pi/6)) * math.sin(n*pi/12*(2*k-1)) * math.sin(n*pi/12) * math.exp(-n**2 * pi**2 * param_D * t)
            
            qvals[j,i] -= (24/11)*term
    
    # print(qvals)
    # exit()

    color = plt.cm.rainbow(np.linspace(0,1,len(timetotal))[::-1])

    ncols = 3
    fig, ax = plt.subplots(math.ceil(len(timetotal)/ncols),ncols,constrained_layout=True,sharex=True,sharey=True)
    for i, axis in enumerate(ax.flatten()):
        if i >= len(timetotal):
            continue
        axis.stairs(qvals[i])
        axis.set_title(f"$t={timetotal[i]}$")
    
        axis.set_xticks([0,1,2,3,4,5,6])
        axis.set_ylim(0,1)
        axis.set_xlim(0,6)

    fig.suptitle(f"The distribution of particles in bins w.r.to time, $D={param_D}$,\nexpanded to $n={numterms}$ terms")
    fig.supxlabel("Bins $k=1,...,6$")
    fig.supylabel("$Q_k(t)$")
    
    fig.savefig("qvalsplot.pdf")
    plt.show()
