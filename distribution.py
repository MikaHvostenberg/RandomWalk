import numpy as np
import matplotlib.pyplot as plt
import math


# computation of Q_k(t)


def main(
        timetotal:np.ndarray,
        display:int,
        param_D:float=0.005,
        numterms:int=200,
        nbins:int=6,
        norm_first:bool=True,
        norm_last:bool=True,
        time_fixed:float=100):
    """
    Main part of the program. Choose which plot to make.
    """
    pi = math.pi

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
            
            if norm_first:
                qvals[j,i] = qvals[j,i]/qvals[j,0] # fix the first bin to be 1

        print(f"Computed bin {k}")
    # print(qvals)
    # exit()

    color = plt.cm.rainbow(np.linspace(0,1,len(timetotal))[::-1])

    if display == 0:
        return qvals
    
    if display == 1:
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

        return None

    if display == 2:
        fig, ax = plt.subplots(1,1,constrained_layout=True, sharex=True,sharey=True)
        for i in range(nbins):
            ax.plot(timetotal,qvals[:,i],label=f"Bin $k={i+1}$")
            # ax.set_title(f"Bin $k={i+1}$")

        ax.legend(loc="upper right")
        fig.supxlabel("$t$")
        fig.supylabel("$Q_k(t)$")
        fig.suptitle(f"The distribution of particles in bins w.r.to time, $D={param_D}$,\nexpanded to $n={numterms}$ terms")
        

        fig.savefig("qvals-per-bin-plot.pdf")

        return None

    if display == 3:
        fig, ax = plt.subplots(1,1,constrained_layout=True, sharex=True,sharey=True)
        for i,qk in enumerate(qvals[::-1]):
            ax.stairs(qk,color=color[i])

        fig.supxlabel("Bin $k=1,...,6$")
        fig.supylabel("$Q_k(t)$")
        fig.suptitle(f"The distribution of particles in bins w.r.to time, $D={param_D}$,\nexpanded to $n={numterms}$ terms")
        

        fig.savefig("qvals-oneplot.pdf")

        return None
    
    if display == 4:
        fig, ax = plt.subplots(1,1,constrained_layout=True, sharex=True,sharey=True)
        ax.plot(timetotal*param_D/time_fixed, qvals[:,-1])
        
        fig.supylabel(f"$Q_6(t={time_fixed})$")
        fig.supxlabel("$D$")
        fig.suptitle(f"Flux depending on $D$ at a fixed $t={time_fixed}$")
        fig.savefig("fluxplot.pdf")

        return None

    else:
        raise ValueError("Incorrectly specified display.")


if __name__ == "__main__":    

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
    # timetotal = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,24,48,96,192,384])
    
    # make sure the t values is hit!
    timetotal = np.linspace(0,100,1001)
    
    main(timetotal,display=4,param_D=0.005,numterms=200,nbins=6,norm_first=True,time_fixed=100)

    plt.show()
    exit()