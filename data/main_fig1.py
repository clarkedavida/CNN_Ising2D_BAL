# 
# main_fig1.py                                                               
# 
# D. Clarke
# 

import numpy as np
from latqcdtools.base.readWrite import readTable
from latqcdtools.base.plotting import plot_dots, plt, set_params, clearPlot, BACKGROUND,\
    FOREGROUND, plot_vspan, plot_hline, plot_fill
from latqcdtools.base.initialize import initialize, finalize
from latqcdtools.base.printErrorBars import get_err_str
from latqcdtools.base.utilities import toNumpy, naturalSort
import latqcdtools.base.logger as logger
from latqcdtools.math.spline import getSplineErr
from latqcdtools.statistics.jackknife import jackknife
from latqcdtools.statistics.statistics import std_mean
from latqcdtools.physics.statisticalPhysics import reweight
import matplotlib.patches as patches


initialize()

NJACK = 40   # Number of jackknife bins
d     = 2    # dimension

# The user decides here whether to plot the prediction susceptibility (CHI) or
# the prediction (M). Also decide which ML model to look at.
#PLOTKIND = 'CHI'
PLOTKIND = 'M'
#MLmodel = 'BAL'
MLmodel = 'BAL20'


# Temperatures
Ts = [ "0.500",  "0.900",  "1.300",  "1.700",  "2.100",  "2.260",  "2.267",  "2.271",  "2.276",  "2.300",  "2.600",  "3.000",  "3.400",
       "0.600",  "1.000",  "1.400",  "1.800",  "2.200",  "2.262",  "2.268",  "2.272",  "2.278",  "2.310",  "2.700",  "3.100",  "3.500",
       "0.700",  "1.100",  "1.500",  "1.900",  "2.240",  "2.264",  "2.269",  "2.273",  "2.280",  "2.400",  "2.800",  "3.200",
       "0.800",  "1.200",  "1.600",  "2.000",  "2.250",  "2.266",  "2.270",  "2.274",  "2.290",  "2.500",  "2.900",  "3.300", ]
Ts = naturalSort(Ts)
Treal = []
for T in Ts:
    Treal.append(float(T))
Treal = np.array(Treal)


def susc(data,temperature=1):
    return std_mean(data**2)/temperature - std_mean(data)**2/temperature

def RWSUSC(data,xRW,x0) -> float:
    """ Reweight the susceptibility. The susceptibility is an observable that is
    defined in terms of expectation values. At the same time, we think of the
    reweight() method as a redefined expectation value.

    Args:
        data (list): a list [M, E] 
        xRW (float): the point we are RWing to 
        x0 (float): the starting point (plays role 1/T)

    Returns:
        float: reweighted susceptibility 
    """
    X = data[0]
    S = data[1]
    return x0*( reweight(X**2,xRW,x0,S) - reweight(X,xRW,x0,S)**2 )

def RWM(data,xRW,x0) -> float:
    """ Reweight the magnetization/prediction """
    X = data[0]
    S = data[1]
    return reweight(X,xRW,x0,S) 

def error(func, dat, args=(), conf_axis=1):
    return jackknife( func, dat, args=args, numb_blocks=NJACK, conf_axis=conf_axis )


dT = {640 : 0.003, 760 : 0.003,}

ymin = {
    'BAL20' : 12000,
    'BAL'   : 4000,
}
xmin = 2.266
xmax = 2.274


if MLmodel=='BAL':
    L = 760 
elif MLmodel=='BAL20':
    L = 640 
else:
    logger.TBError('Unknown ML model',MLmodel)



Mavg, Merr, chiM, chiMerr = [], [], [], []

V = L**d

imax   = -1
chimax = -1
for i,T in enumerate(Ts):

    confs_M, _, Es  = readTable(f'MCMC/{L}/{T}.d',dtype="U11,f8,f8")
    confs_P, Ps     = readTable(f'ML/{MLmodel}_{L}/{T}.d',dtype="U11,f8")

    # For RWing of the prediction we have to make sure we measure E and P on the same config
    for iconf in range(len(confs_M)):
        if confs_M[iconf]!=confs_P[iconf]:
            logger.info('problem at iconf=',iconf)
            logger.TBError(f'MCMC ordering is incorrect for L={L}, T={T}')

    MJm, MJe = error( std_mean, Ps ) 
    Mavg.append(MJm)
    Merr.append(MJe)

    chi, chie = error(susc, Ps, args=(float(T),)) 
    chiM.append(V*chi)
    chiMerr.append(V*chie)

    # Figure out where the max is. We also want to save the Es and Ms for this max point, because this max point will
    # serve as our starting point for the reweighting. The idea of reweighting is that we rescale our measurements 
    # according to exp(- change in S as you change T to the RW point). This is equivalent to rescaling (reweighting) 
    # the measurement by the likelihood of having a configuration with action SRW (as opposed to the original action) 
    if chi > chimax:
        chimax = chi
        imax   = i
        E0     = -V*Es
        M0     = Ps
        if imax > len(Ts):
            logger.TBError('Something went weird with imax determination, imax=',i)


# Plot simulation points
if PLOTKIND=='CHI':
    plot_dots(Treal, chiM, chiMerr, color='blue',marker=None)
elif PLOTKIND == 'M':
    plot_dots(Treal, Mavg, Merr, color='blue',marker=None)
else:
    logger.TBError('Invalid plotkind',PLOTKIND)
logger.info('L, imax=',L,imax)


Tfinals, Tefinals, chifinals, chiefinals = [], [], [], []

RWindices = [imax-1, imax, imax+1]


for iRW in RWindices: 

    Tlo  = Treal[iRW] - dT[L]
    Thi  = Treal[iRW] + dT[L]
    TRWs = np.linspace(Tlo,Thi,90)
    Tstr = "{:.3f}".format(Treal[iRW])

    confs_M, _, Es  = readTable(f'MCMC/{L}/{Tstr}.d',dtype="U11,f8,f8")
    confs_P, Ps     = readTable(f'ML/{MLmodel}_{L}/{Tstr}.d',dtype="U11,f8")

    E0 = -V*Es
    T0 = Treal[iRW]
    M0 = Ps

    chi , chie = error(susc, Ps, args=(Treal[iRW],) ) 
    Msim, Merr = error(std_mean,M0)

    if PLOTKIND=='CHI':
        plot_dots([T0],[V*chi],[V*chie],color='red',marker=None,ZOD=FOREGROUND)
    elif PLOTKIND=='M':
        plot_dots([T0],[Msim],[Merr],color='red',marker=None,ZOD=FOREGROUND)

    if len(M0)!=len(E0):
        logger.TBError('M0 and E0 have different shapes. L=',L)

    # RW from T0 to each TRW in the vicinity around TRW
    chiRW, chiRWe, MRW, MRWe = [], [], [], []
    for TRW in TRWs:
        RW, RWe = error( RWSUSC, [M0,E0], args=(1/TRW,1/T0), conf_axis=1 ) 
        chiRW.append(V*RW)
        chiRWe.append(V*RWe)
        RW, RWe = error( RWM, [M0,E0], args=(1/TRW,1/T0), conf_axis=1 ) 
        MRW.append(RW)
        MRWe.append(RWe)
    chiRW, chiRWe, MRW, MRWe = toNumpy(chiRW, chiRWe, MRW, MRWe)

    jmax = np.argmax(chiRW)
    chiefinals.append(chiRWe[jmax])

    def findTmax(data) -> float:
        SUSCmax = -1
        tempmax = -1
        for TRW2 in TRWs:
            SUSCRWm = RWSUSC(data,1/TRW2,1/T0)
            if SUSCRWm > SUSCmax:
                SUSCmax = SUSCRWm
                tempmax = TRW2
        if tempmax==-1:
            logger.TBRaise('Failed to locate maximum') 
        return tempmax

    _, temp_err = error( findTmax, [M0,E0], conf_axis=1)
    Tefinals.append(temp_err) 

    tspl=np.linspace(TRWs[0],TRWs[-1],301)
    if PLOTKIND=='CHI':
        center, err = getSplineErr(TRWs,tspl,chiRW,chiRWe,natural=True)
        plot_fill(tspl,center,err,color='grey',marker=None,ZOD=BACKGROUND,alpha=0.2,center=False) 
    elif PLOTKIND=='M':
        center, err = getSplineErr(TRWs,tspl,MRW,MRWe,natural=True)
        plot_fill(tspl,center,err,color='grey',marker=None,ZOD=BACKGROUND,alpha=0.2,center=False) 

    # Will be used to estimate systematic error
    Tfinals.append(TRWs[jmax]) 
    chifinals.append(chiRW[jmax])


# Add systematic in quadrature to statistical error computed from the jackknife carried out
# starting at the naive maximum temperature (which should be closest to the 'true' max)
syst    = (np.max(Tfinals)  -np.min(Tfinals))/2
systchi = (np.max(chifinals)-np.min(chifinals))/2

Tc  = std_mean(Tfinals)
TRWerr = np.max(Tefinals)
Tce = np.sqrt( TRWerr**2 + syst**2 )

plot_vspan(minVal=Tc-Tce, maxVal=Tc+Tce, color='yellow', alpha=0.3, label='$T_c(L)$',ZOD=BACKGROUND)

Tstr="{:.3f}".format(Treal[imax])

# Now that we have Tc, we need to reweight the magnetization to Tc to get m(Tc)
confs_M, _, Es = readTable(f'MCMC/{L}/{Tstr}.d',dtype="U11,f8,f8")
confs_P, M0    = readTable(f'ML/{MLmodel}_{L}/{Tstr}.d',dtype="U11,f8")
E0 = -V*Es
T0 = Treal[imax]

chiTc  = std_mean(chifinals)
chiTce = np.max(chiefinals)

def M_at_Tc(temp):
    return error(RWM,[M0,E0],args=(1/temp,1/T0),conf_axis=1)

Tleft  = Tc-Tce
Tright = Tc+Tce
Mleft , Mlefterr = M_at_Tc(Tleft)
Mright, Mrighterr = M_at_Tc(Tright)

MTc   = std_mean([Mleft,Mright])
MTce  = np.max([Mlefterr,Mrighterr])
systM = np.abs(Mright-Mleft)/2

chi_errTOT = np.sqrt(chiTce**2+systchi**2)
M_errTOT   = np.sqrt(MTce**2  +systM**2)


# Box that shows the final error in magnetization or susceptibility
if PLOTKIND=='CHI':
    rect = patches.Rectangle((Tc-Tce, chiTc-chi_errTOT), 2*Tce, 2*chi_errTOT, linewidth=1, edgecolor=None, facecolor='red',alpha=0.2,zorder=BACKGROUND)
elif PLOTKIND=='M':
    rect = patches.Rectangle((Tc-Tce, MTc-M_errTOT), 2*Tce, 2*M_errTOT, linewidth=1, edgecolor=None, facecolor='red',alpha=0.2,zorder=BACKGROUND)
    plot_hline(y=0.5,color='red',linestyle='dotted',ZOD=BACKGROUND)
ax=plt.gca()
ax.add_patch(rect)


if PLOTKIND=='CHI':
    set_params(ylabel='$\\chi_P$',xlabel='$T$',title=f'{MLmodel}, $T_c({L})={get_err_str(Tc,Tce)}$',
               xmin=xmin,xmax=xmax,ymin=ymin[MLmodel])
elif PLOTKIND=='M':
    set_params(ylabel='$\\langle P\\rangle$',xlabel='$T$',title=f'{MLmodel}, $T_c({L})={get_err_str(Tc,Tce)}$',
               xmin=xmin,xmax=xmax)


plt.savefig('fig1.pdf')
plt.show()
clearPlot()
finalize()
