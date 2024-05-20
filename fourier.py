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
        
        # initial condition LAMBDA(x) = 11/12
        I_lam = 11/(12*n*math.pi)*(1-math.cos(n*math.pi/6))
        
        # initial condition LAMBDA(x) = -(11/8)*12^2 (x-1/12)^2 + 11/8 
        # I_lam = -24/(n*math.pi)**3*(-12+12*math.cos(n*math.pi/6)+n*math.pi*math.sin(n*math.pi/6))
        
        I_s = 1/(n*math.pi)
        coeffs.append(2*(I_lam - I_s))
    
    #small list! not the axis
    timetotal = np.concatenate(
        (
            np.linspace(0,1,40),
            np.linspace(1,5,80),
            np.linspace(5,10,40),
            np.linspace(10,20,40),
            np.linspace(20,50,80),
            np.linspace(50,100,40),
        ),
        axis=None
    )

    xrange = np.linspace(0,1,1000)
    param_D = 0.005

    color = plt.cm.rainbow(np.linspace(0,1,len(timetotal))[::-1])

    fig, ax = plt.subplots(1,1,constrained_layout=True)
    # ax.set_ylim(bottom=0,top=6)
    # ax.set_xlim(left=0,right=1)
    ax.set_xticks([0,1/6,2/6,3/6,4/6,5/6,1],labels=["0","1/6","2/6","3/6","4/6","5/6","1"])
    ax.set_title(f"$T(x,t)$ for $0<t<100$, $D= {param_D}$, expanded to {numterms} terms\nwith $\Lambda(x)=11/12$ over $[0,1/6]$")    
    ax.set_xlabel("$x$")
    ax.set_ylabel("$T(x,t)$")


    for j,t in enumerate(timetotal[::-1]):
        solvals = 1-xrange
        for i, cf in enumerate(coeffs):
            solvals += cf*np.sin((i+1)*np.pi*xrange)*np.exp(-(i+1)**2*np.pi**2*param_D*t)

        ax.plot(xrange,solvals,c=color[j])
    
    fig.savefig("initialplot-parabolic.pdf")
    plt.show()
