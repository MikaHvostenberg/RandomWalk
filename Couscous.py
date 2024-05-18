import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare

def f(x,a,b):
    return a*x+b
def chi_sq(dat,mod):
    return np.sum((dat-mod)**2)


data = pd.read_csv('/home/edo/Downloads/Couscous.csv',delimiter = ',')
fig,axs= plt.subplots(12,6,figsize=(10,10),constrained_layout=True, sharex=True, sharey=True)
chis=np.zeros((12,6))
amplitudes=[0.5,0.6,0.7,0.8,0.9,0.98]
for id, j in enumerate (amplitudes):
    u=data.loc[data['Amplitude']==j]
    l=len(np.array(u))
    print(l)
    ww=False
    for i in range(l):
        x=np.array(range(1,7))  
        y=np.array(u.iloc[i,[9,3,4,5,6,8]]) 
        t=np.array(u['Time'])
        print(t)
        axs[i,id].bar(x,y)
        pars,_=curve_fit(f,x,y)
        axs[i,id].plot(x,f(x,*pars),color='r')
        axs[i,id].set_ylim(0,1)
        axs[i,id].tick_params(right = False , labelbottom = False, bottom=False)
        chis[i,id]=chi_sq(y,f(x,*pars))
        if ww==True:
            axs[i,id].set_title(f't={t[i]}s')
            if id==0:
                axs[i,id].set_ylabel('m [g]')
        else:
            axs[i,id].set_title(f'Amplitude {j} A')
            ww=True
print(chis)
fig.delaxes(axs[8,2])
fig.delaxes(axs[9,2])
for u in (10,11):
    fig.delaxes(axs[u,0])
    fig.delaxes(axs[u,5])
    fig.delaxes(axs[u,2])
fig.suptitle('Time development of weight in each compartment',size=15)

figa,axsa=plt.subplots(6,figsize=(8,10),constrained_layout=True)
for k,a in enumerate (amplitudes):
    u=data.loc[data['Amplitude']==a]
    t=np.array(u['Time'])
    ch=np.trim_zeros(np.array(chis[:,k]), trim='b')
    print(len(t),len(ch))
    axsa[k].plot(t,ch)
    axsa[k].set_title(f'Amplitude {a} A')
    axsa[k].set_xlabel('t [s]')
figa.suptitle('Chi squared of a linear fit of weights versus time',size=15)
plt.show()
