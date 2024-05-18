import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    coeffs = []
    numterms = 1000

    for i in range(numterms):
        n = i+1
        # coeffs.append(2/(n*math.pi)*(5-6*math.sin(n*math.pi/6)+math.cos(n*math.pi)))
        # coeffs.append(2/(n*math.pi)*(6-6*math.sin(n*math.pi/6)))
        
        # initial condition LAMBDA(x) = -6*12^2 (x-1/12)^2 + 6 
        I_lam = -24/(n*math.pi)**3*(-12+12*math.cos(n*math.pi/6)+n*math.pi*math.sin(n*math.pi/6))
        I_s = 1/(n*math.pi)
        coeffs.append(2*(I_lam - I_s))
    
    timerangeone = np.linspace(0,2,40) #small list! not the axis
    timerangetwo = np.linspace(2,5,40)
    timerangethree = np.linspace(5,20,80)
    timerangefour = np.linspace(20,100,40)
    timetotal = np.concatenate((timerangeone,timerangetwo,timerangethree,timerangefour),axis=None)

    xrange = np.linspace(0,1,1000)
    param_D = 0.005

    color = plt.cm.rainbow(np.linspace(0,1,len(timetotal))[::-1])

    fig, ax = plt.subplots(1,1,constrained_layout=True)
    # ax.set_ylim(bottom=0,top=6)
    # ax.set_xlim(left=0,right=1)
    ax.set_xticks([0,1/6,2/6,3/6,4/6,5/6,1],labels=["0","1/6","2/6","3/6","4/6","5/6","1"])
    ax.set_title("$T(x,t)$ for $0<t<100$, $D=$" + f"{param_D}, expanded to {numterms} terms")    
    ax.set_xlabel("$x$")
    ax.set_ylabel("$T(x,t)$")


    for j,t in enumerate(timetotal[::-1]):
        solvals = 1-xrange
        for i, cf in enumerate(coeffs):
            solvals += cf*np.sin((i+1)*np.pi*xrange)*np.exp(-(i+1)**2*np.pi**2*param_D*t)

        ax.plot(xrange,solvals,c=color[j])
    
    fig.savefig("Tplot-parabolic.pdf")
    plt.show()
