
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

def stationary_flow(N:int,particles:int,steps:int,weights:list,graph=True):
    '''''This function simulates the stationary flow through the probabilistic setup and calculates
    the average heat flow (over the last 100 seconds)'''''
     #here the key parameters are defined
    walks=[-1,0,1]
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
    #printing block(makes a graph of a single measurement)
    if graph==True:
        plt.plot(flow,color="black",label="momentary particle flux")
        plt.grid()
        plt.axhline(y=avg_flow,color='red',linestyle='dashed',label=f"running average: {avg_flow} ")
        plt.legend()
        plt.show()
    return avg_flow
#draws a single thing
#stationary_flow(10,10**5,200,[0.25,0.5,0.25])

#this is another experimental function
def measure_flow(N=10):
    """"Fucntion that measures the stable heat flow as a function of the probabilities of 
    particles going left or right"""
    #target array
    steady_flow=[]
    probabilities=np.linspace(0,0.5,N)
    for prob in probabilities:
        steady_flow.append(stationary_flow(10,10**5,200,[prob,1-2*prob,prob],graph=False))
    plt.plot(probabilities,steady_flow)
    plt.show()
    return
#lets make a graph of the flow as a function of probabilities
measure_flow()
