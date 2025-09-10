import matplotlib.pyplot as plt
import numpy as np

def plotData(
    minimizationResult, 
    filename, 
    fieldNames
):
    
    if minimizationResult["failureReason"]:
        return
    markers = ['^', 'v', 'o', 's', 'D', '<', '>', 'p', '*', 'h']
    tempList = minimizationResult["T"]

    for idx, vev in enumerate(minimizationResult["vevLocation"]/np.sqrt(tempList)):
        plt.plot(
            tempList, 
            vev, 
            label=f"{fieldNames[idx]}", 
            linestyle='None', 
            marker=markers[idx],
            markersize=3.5
        )
        
    plt.legend(loc="best")
    plt.ylabel(r"$\dfrac{v}{\sqrt{T}}$", rotation=0, labelpad=10)
    plt.xlabel("T (GeV)")
    plt.savefig(f"{filename}.png")
    plt.close()    
    return
