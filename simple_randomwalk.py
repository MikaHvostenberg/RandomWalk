
import matplotlib.pyplot as plt
from matplotlib import cm
import random 
import numpy as np

#parameters of the simulation
domains=50
particles=10**4


def statistics(N:int,particles:int,steps:int):
    #here the key parameters are defined
    walks=[-1,0,1]
    weights=[0.4,0.2,0.4]
    #here we define the initial destribution
    current=np.zeros(N)
    current[0]=particles
    for step in range(0,steps,1):
        tmp=np.zeros_like(current)
        for j in range(0,len(tmp),1):
            for k in range(0,int(current[j])):
                if j==0:
                    walk=random.choices([0,1],[weights[0]+weights[1],weights[2]])
                elif j==len(current)-1:
                    walk=random.choices([0,-1],[weights[0]+weights[1],weights[2]])
                else:
                    walk=random.choices(walks,weights)
                tmp[j+int(walk[0])]+=1
        current=tmp
    final=current
    print(np.sum(final))
    print(final)
    my_cmap = plt.get_cmap("inferno")
    rescale = lambda final: (final - np.min(final)) / (np.max(final) - np.min(final))
    # Create bar chart
    plt.bar(np.arange(1,domains+1,1),final, color=my_cmap(rescale(final)))
    plt.show()
    return current

#statistics(domains,particles,750)
#here is the function for the stationary flow of the particles

def stationary_flow(N:int,particles:int,steps:int):
     #here the key parameters are defined
    walks=[-1,0,1]
    weights=[0.25,0.5,0.25]
    flow=np.zeros(steps)
    #here we define the initial destribution
    current=np.zeros(N)
    current[0]=particles
    for step in range(0,steps,1):
        tmp=np.zeros_like(current)
        for j in range(0,len(tmp),1):
            if j==len(current)-1:
                flow[step]=current[j]
            for k in range(0,int(current[j])):
                if j==0:
                    walk=random.choices([0,1],[weights[0]+weights[1],weights[2]])
                elif j==len(current)-1:
                    walk=[1-len(current)]
                else:
                    walk=random.choices(walks,weights)
                tmp[j+int(walk[0])]+=1
        current=tmp
    avg_flow=np.average(flow[-100:])
    print(avg_flow)
    plt.plot(flow)
    plt.grid()
    plt.show()
    return current

stationary_flow(10,10**5,300)
