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
        norm_last:bool=True):
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
        fig2, ax2 = plt.subplots(1,1,constrained_layout=True, sharex=True,sharey=True)
        for i in range(nbins):
            ax2.plot(timetotal,qvals[:,i],label=f"Bin $k={i+1}$")
            # ax2.set_title(f"Bin $k={i+1}$")

        ax2.legend(loc="upper right")
        fig2.supxlabel("$t$")
        fig2.supylabel("$Q_k(t)$")
        fig2.suptitle(f"The distribution of particles in bins w.r.to time, $D={param_D}$,\nexpanded to $n={numterms}$ terms")
        

        fig2.savefig("qvals-per-bin-plot.pdf")

        return None

    if display == 3:
        fig3, ax3 = plt.subplots(1,1,constrained_layout=True, sharex=True,sharey=True)
        for i,qk in enumerate(qvals[::-1]):
            ax3.stairs(qk,color=color[i])

        fig3.supxlabel("Bin $k=1,...,6$")
        fig3.supylabel("$Q_k(t)$")
        fig3.suptitle(f"The distribution of particles in bins w.r.to time, $D={param_D}$,\nexpanded to $n={numterms}$ terms")
        

        fig3.savefig("qvals-oneplot.pdf")

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
    timetotal = np.linspace(0,100,1000)
    
    main(timetotal,display=2,param_D=0.005,numterms=200,nbins=6,norm_first=False)

    plt.show()
    exit()