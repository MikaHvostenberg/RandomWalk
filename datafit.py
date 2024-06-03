import numpy as np 
import math
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Callable
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
        axis.set_title(f"$t={tvals[i]}$ s")

    fig.supxlabel("Bin")
    fig.supylabel("Normalised amount, arb. units")
    fig.suptitle(f"Experimental and theoretical $Q_k$ values with $D={dval} \\pm {derr}$ Hz length at $I={ampl}$ A")
    fig.savefig("plots/" + f"ampl{ampl}" + savetitle + ".pdf")

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


def make_normal_qvals(data:np.ndarray):
    """
    Makes the normalised concatenated qvals for fitting
    and qvals for plotting.
    """
    # make normalised q values for amplitude
    normalisation = data[:,2]
    qvals = np.concatenate(
        [
        data[:,2]/normalisation, 
        data[:,3]/normalisation, 
        data[:,4]/normalisation, 
        data[:,5]/normalisation, 
        data[:,6]/normalisation, 
        data[:,8]/normalisation #flux 8, zero 7
        ]
    ) 
    
    # make nonconcatenated q values as for plotting
    databefore = data[:, [2,3,4,5,6,8]]
    qvals_forplotting = databefore/np.transpose(np.tile(normalisation, (6,1)))

    return qvals, qvals_forplotting



def compute_fit(
        funct:Callable, 
        tvals:np.ndarray, 
        qvalsexp:np.ndarray, 
        p0:float=0.0003, 
        bounds:list[float]=[0.0001,0.0005],
        compfitname:str=""
        ) -> tuple[float, float, float, float]:

    # compute fitting
    tvals_concat = np.tile(tvals,6)
    params, pcovs = sp.optimize.curve_fit(funct, tvals_concat, qvalsexp, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcovs))[0]
    dparam = params[0]
    
    print(f"Executing compute_fit {compfitname}")
    print(f"D={params} +- {perr} Hz length")
    
    rnd_perr, dig_p = rnd_fl(perr) 
    rnd_p = round(params[0], dig_p)

    return dparam, perr, rnd_p, rnd_perr


def funcexp(x, a, c):
    return a*np.exp(c*x)


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
    # print(tvallist)


    # subtract the flux
    for i in range(len(datalist)):
        #dummy variables, keep the time and 0 columns the same
        ti = datalist[i][:,1]
        q60 = datalist[i][:,7]
        qi = datalist[i] 
        x = datalist[i][:,8][:, np.newaxis]
        datalist[i] = np.abs(qi - x)
        datalist[i][:,1] = ti
        datalist[i][:,7] = q60
        # print(datalist[i])


    
    # execude code for each data
    cont = [0,0,0,0,0,0,0] # select which to fit (1=fit, 0=skip)
    ampls = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.98]
    p0s = [0.0001,0.0003,0.0015,0.02,0.03,0.04,0.04]
    boundvals = [
        [0.00005, 0.0002],
        [0.00015, 0.0015],
        [0.0007, 1],
        [0.0009, 1],
        [0.0002, 1],
        [0.0005, 1],
        [0.0005, 1],
    ]
    for i in range(len(tvallist)):
        if cont[i] == 0:
            print(f"Skipping amplitude {ampls[i]}")
            # plt.figure()
            # plt.savefig(f"plots/ampl{ampls[i]}.pdf")
            continue
        tvals = tvallist[i]
        qvals, qvals_forplotting = make_normal_qvals(datalist[i])
        dpar, perr, rnd_d, rnd_perr = compute_fit(function_to_fit, tvals, qvals, p0=p0s[i], bounds=boundvals[i], compfitname=str(ampls[i]))
        qtheor = compute_qvals(tvals, param_D=dpar)

        plot_comparison(tvals, qvals_forplotting, qtheor, ampls[i], rnd_d, rnd_perr, "")
    
    length_unit = 20 # centimeters
    arr_ampls = np.array(ampls)
    dfitted = np.array([0.00005, 0.00015, 0.00016, 0.0007, 0.0002, 0.0004, 0.0005])*length_unit**2 # Hz cm^2
    dfittederr = np.array([0.00005, 0.00002, 0.00003, 0.0002, 0.00006, 0.0001, 0.0001])*length_unit**2 # Hz cm^2
    
    fig, ax = plt.subplots(1,1, constrained_layout=True, sharex=True, sharey=True)

    eparams, epcovs = sp.optimize.curve_fit(funcexp, arr_ampls, dfitted, sigma=dfittederr)
    eerrs = np.sqrt(np.diag(epcovs))
    rnd_erra, dig_pa = rnd_fl(eerrs[0]) 
    rnd_errc, dig_pc = rnd_fl(eerrs[1]) 
    rnd_pa = round(eparams[0], dig_pa)
    rnd_pc = round(eparams[1], dig_pc)


    xvals = np.linspace(np.min(arr_ampls), np.max(arr_ampls), 1000)
    yvals = funcexp(xvals, eparams[0], eparams[1])
    yerror = funcexp(xvals, eerrs[0], eerrs[1])
    yminus = funcexp(xvals, eparams[0]-eerrs[0], eparams[1]-eerrs[1])
    yplus = funcexp(xvals, eparams[0]+eerrs[0], eparams[1]+eerrs[1])

    ax.plot(xvals, yvals, c="#336ea0", label="$D=ae^{cI}$, where " + f"$a={rnd_pa}\\pm {rnd_erra}$, $c={rnd_pc} \\pm {rnd_errc}$")
    ax.fill_between(xvals, yminus, yplus, alpha=0.5, facecolor="#dde9f4", edgecolor="#74a7d2")

    ax.errorbar(arr_ampls, dfitted, yerr=dfittederr, xerr=0.01*np.ones_like(arr_ampls), ecolor="black", color='r', linestyle='', marker='.')
    ax.set_yscale("log")
    ax.legend(loc="lower right")
    fig.supxlabel("$I$, A")
    fig.supylabel("$D$, Hz cm$^2$")
    fig.savefig("plots/diplot.pdf")

    exit()


