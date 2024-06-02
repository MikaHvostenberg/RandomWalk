import numpy as np 
import math
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
from distribution import compute_qvals

def compute_qk(
        timevalue:np.ndarray,
        param_D:float,
        bin_k:int,
        numterms:int=200,
        norm_first:bool=True,
    ):
    """
    Computes the expected Q value in box k.
    """
    pi = math.pi
    # iterate over each of the bins and then for each time value compute the series

    qvalue = 0

    # bin index i and k in formula
    qvalue += (13-2*bin_k)/11

    term = 0
    for n in range(1,numterms+1):
        term += 1/(n**2*pi**2) * (1+11*math.cos(n*pi/6)) * math.sin(n*pi/12*(2*bin_k-1)) * math.sin(n*pi/12) * np.exp(-n**2 * pi**2 * param_D * timevalue)

    qvalue -= (24/11)*term
    
    # perform normalisation by the first value
    if norm_first:
        qfirst = 1
        term = 0
        for n in range(1,numterms+1):
            term += 1/(n**2*pi**2) * (1+11*math.cos(n*pi/6)) * math.sin(n*pi/12*(2*bin_k-1)) * math.sin(n*pi/12) * np.exp(-n**2 * pi**2 * param_D * timevalue)

        qfirst -= (24/11)*term

        qvalue = qvalue/qfirst # fix the first bin to be 1

    return qvalue


def function_to_fit(time_combined:np.ndarray, param_d:float):
    """
    Computes the function for fitting.
    """

    t1, t2, t3, t4, t5, t6 = np.hsplit(time_combined,6)

    result_k1 = compute_qk(t1,param_D=param_d,bin_k=1)
    result_k2 = compute_qk(t2,param_D=param_d,bin_k=2)
    result_k3 = compute_qk(t3,param_D=param_d,bin_k=3)
    result_k4 = compute_qk(t4,param_D=param_d,bin_k=4)
    result_k5 = compute_qk(t5,param_D=param_d,bin_k=5)
    result_k6 = compute_qk(t6,param_D=param_d,bin_k=6)

    return np.concatenate([result_k1,result_k2,result_k3,result_k4,result_k5,result_k6])


def plot_comparison(
        tvals:np.ndarray, 
        qexperlist:np.ndarray, 
        qtheorlist:np.ndarray, 
        ampl:float, 
        dval:float, 
        derr:float, 
        savetitle:str
    ):
    """
    Plots the experimental q values against the fitted q values.
    """

    nrows = int(math.ceil(len(tvals)/3))
    fig, ax = plt.subplots(nrows, 3, figsize=(10,19), constrained_layout=True, sharex=True, sharey=True)

    for i, axis in enumerate(ax.flatten()):
        if i == len(tvals):
            break
        
        qexper = qexperlist[i]
        qtheor = qtheorlist[i]

        axis.stairs(qexper, color="r", label="Exp.")
        axis.stairs(qtheor, color="b", label="Thr.")
        axis.legend(loc="upper right")
        axis.set_title(f"$t={tvals[i]}$")

    fig.supxlabel("Bin")
    fig.supylabel("Normalised amount, arb. units")
    fig.suptitle(f"Experimental and theoretical $Q_k$ values with $D={dval} \\pm {derr}$ at $I={ampl}$ A")
    fig.savefig("plots/" + savetitle + ".pdf")

    return None


def rnd_fl(num):
    """
    Round the float to the first non-zero digit.
    """
    if not num:
        return num, 0
    current_num = abs(num) * 10
    round_value = 1

    while not (current_num//1):
        current_num *= 10
        round_value +=1

    return round(num, round_value), round_value



if __name__ == "__main__":
    df = pd.read_csv("datafile.csv")
    # print(df)

    data = df.to_numpy()
    data = data[8:]

    ampl5  = data[np.where(data == 0.5)[0]]
    ampl6  = data[np.where(data == 0.6)[0]]
    ampl7  = data[np.where(data == 0.7)[0]]
    ampl75 = data[np.where(data == 0.75)[0]]
    ampl8  = data[np.where(data == 0.8)[0]]
    ampl9  = data[np.where(data == 0.9)[0]]
    ampl98 = data[np.where(data == 0.98)[0]]
    
    tvallist = []
    datalist = [ampl5, ampl6, ampl7, ampl75, ampl8, ampl9, ampl98]

    # make a list of all t values to try, seconds
    for i in range(len(datalist)): 
        tvallist.append(datalist[i][:,1])
    print(tvallist)

    # make normalised q values for amplitude 0.5
    normalisation_5 = datalist[0][:,9]
    qvals_5 = np.concatenate(
        [
        datalist[0][:,9]/normalisation_5, 
        datalist[0][:,3]/normalisation_5, 
        datalist[0][:,4]/normalisation_5, 
        datalist[0][:,5]/normalisation_5, 
        datalist[0][:,6]/normalisation_5, 
        datalist[0][:,8]/normalisation_5 #flux
        ]
    ) # normalise by the first bin
    databefore = datalist[0][:, [9,3,4,5,6,8]]
    print(np.transpose(np.tile(normalisation_5, (6,1))))
    print(databefore)
    qvals_5_forplotting = databefore/np.transpose(np.tile(normalisation_5, (6,1)))
    print(qvals_5_forplotting)
    print(qvals_5)

    # compute fitting
    tvals_5 = np.tile(tvallist[0],6)
    params5, pcovs5 = sp.optimize.curve_fit(function_to_fit, tvals_5, qvals_5, p0=[0.0003], bounds=[0.0001,0.0005])
    perr5 = np.sqrt(np.diag(pcovs5))[0]
    
    print(f"D={params5} +- {perr5} Hz length")
    
    qtheor5 = compute_qvals(tvallist[0],param_D=params5[0])


    rnd_perr5, dig_p5 = rnd_fl(perr5) 
    rnd_p5 = round(params5[0], dig_p5)
    plot_comparison(tvallist[0], qvals_5_forplotting, qtheor5, 0.5, rnd_p5, rnd_perr5, "ampl0.5")





