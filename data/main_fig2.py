# 
# main_fig2.py
# 
# D. Clarke
# 

import numpy as np
from latqcdtools.base.readWrite import readTable
from latqcdtools.base.plotting import plt, set_params, clearPlot, plot_dots, \
    plot_lines, saveFigure, plot_vline
from latqcdtools.base.printErrorBars import get_err_str
from latqcdtools.base.initialize import initialize, finalize
from latqcdtools.base.cleanData import clipRange
from latqcdtools.base.utilities import toNumpy
import latqcdtools.base.logger as logger
from latqcdtools.statistics.statistics import gaudif, error_prop, BAIC, modelAverage, getModelWeights
from latqcdtools.statistics.fitting import Fitter
from latqcdtools.physics.statisticalPhysics import Z2_2d

initialize()
d=2

univ   = Z2_2d()
TcLIT  = 2/np.log(1+np.sqrt(2))
TcLITe = 1e-12
minL   = 65 

def linearFit(x,p):
    return p[0] + p[1]*x

def nuTcFit(x,p):
    return -p[0]*np.log(np.abs(x-p[1])) 


#
# nu, Tc from ML
#
L, TcL, TcLerr, chiL, chiLerr, MTcL, MTcLerr = readTable('fig2_BAL20.d')
ydata = np.log(L)
from scipy.odr import Model, RealData, ODR
def myFit(p,x):
    p0, p1 = p
    return -p0*np.log(np.abs(x-p1)) 
linear_model = Model(myFit)
data = RealData(TcL, ydata, sx=TcLerr, sy=1e-5*ydata)
odr = ODR(data, linear_model, beta0=[univ.nu, TcLIT])
output = odr.run()
res = output.beta
res_err = output.sd_beta
quality = output.res_var
quality = "{:.3f}".format(quality)
TcML   = res[1]
TcMLe  = res_err[1]
nuML   = res[0]
nuMLe  = res_err[0]
logger.info('Tc,nu fit quality:',quality)
plot_dots(xdata=TcL,ydata=1/L,xedata=TcLerr,color='blue',label='BAL20',marker='s')
plot_dots(xdata=[TcML],ydata=[0],xedata=[TcMLe],color='blue',marker='*',markerfill=True)
def plotFit(x):
    return np.exp(-nuTcFit(x,[nuML,TcML]) )
xplot=np.linspace(TcML+1e-6,TcL[0]+TcLerr[0],301)
plot_lines(xplot,plotFit(xplot),color='blue',marker=None)


logger.info('   estimated Tc =',get_err_str(TcML, TcMLe))
logger.info('   estimated nu =',get_err_str(nuML, nuMLe))
logger.info(' q(est vs real) =',round( gaudif( TcLIT, TcLITe, TcML, TcMLe)   ,2))


L, TcL, TcLerr, chiL, chiLerr, MTcL, MTcLerr = readTable('fig2_BAL.d')
ydata = np.log(L)
from scipy.odr import Model, RealData, ODR
def myFit(p,x):
    p0, p1 = p
    return -p0*np.log(np.abs(x-p1)) 
linear_model = Model(myFit)
data = RealData(TcL, ydata, sx=TcLerr, sy=1e-5*ydata)
odr = ODR(data, linear_model, beta0=[univ.nu, TcLIT])
output = odr.run()
res = output.beta
res_err = output.sd_beta
quality = output.res_var
quality = "{:.3f}".format(quality)
TcML2   = res[1]
TcMLe2  = res_err[1]
nuML2   = res[0]
nuMLe2  = res_err[0]
logger.info('Tc,nu fit quality:',quality)
plot_dots(xdata=TcL,ydata=1/L,xedata=TcLerr,color='purple',label='BAL',marker='o')
plot_dots(xdata=[TcML2],ydata=[0],xedata=[TcMLe2],color='purple',marker='*',markerfill=True)
def plotFit(x):
    return np.exp(-nuTcFit(x,[nuML2,TcML2]) )
xplot=np.linspace(TcML2+1e-6,TcL[0]+TcLerr[0],301)
plot_lines(xplot,plotFit(xplot),color='purple',marker=None)

logger.info('   estimated Tc =',get_err_str(TcML2, TcMLe2))
logger.info('   estimated nu =',get_err_str(nuML2, nuMLe2))
logger.info(' q(est vs real) =',round( gaudif( TcLIT, TcLITe, TcML2, TcMLe2)   ,2))

plot_vline(TcLIT,color='grey',linestyle='dotted',alpha=0.5)
set_params(xlabel='$T$',ylabel='$1/L$')
saveFigure('nu_Tc.pdf')
plt.show()
clearPlot()




#
# gamma from ML
#

L, TcL, TcLerr, chiL, chiLerr, MTcL, MTcLerr = readTable('fig2_BAL20.d')
ydata, edata = error_prop(np.log,chiL,chiLerr)
data = clipRange(np.vstack((L,ydata,edata)),0,minVal=minL)
ydata, edata = data[1], data[2]
xdata = np.log(data[0])
fit = Fitter(linearFit,xdata,ydata,edata)
res, res_err, chidof = fit.try_fit(algorithms=['curve_fit'],start_params=[1,univ.gamma/univ.nu],show_results=True)
fit.plot_fit(color='blue',alpha=0.2)
fit.plot_data(color='blue',marker='s',label='BAL20')


# Try a BMA here
ICs, gams, gamerrs = [], [], []
for Ncut in [0,1,2,3,4]:
    xcut   = xdata[Ncut:]
    ycut   = ydata[Ncut:]
    ecut   = edata[Ncut:]
    fitcut = Fitter(linearFit,xcut,ycut,ecut)
    rescut, reserrcut, _ = fitcut.try_fit(algorithms=['curve_fit'],start_params=[1,univ.gamma/univ.nu])
    IC = BAIC(xdata=xcut, ydata=ycut, cov=np.diag(ecut), func=linearFit, params=rescut, Ncut=Ncut)
    ICs.append(IC)
    gams.append(rescut[1])
    gamerrs.append(reserrcut[1])
ICs, gams, gamerrs = toNumpy(ICs, gams, gamerrs) 
gamBMA, gamBMAe = modelAverage(gams, gamerrs, ICs)
PrMD = getModelWeights(ICs)
for Ncut in [0,1,2,3,4]:
    xcut   = xdata[Ncut:]
    ycut   = ydata[Ncut:]
    ecut   = edata[Ncut:]
    fitcut = Fitter(linearFit,xcut,ycut,ecut)
    rescut, reserrcut, _ = fitcut.try_fit(algorithms=['curve_fit'],start_params=[1,univ.gamma/univ.nu])
    fitcut.plot_fit(color='blue',linestyle='dashed',alpha=PrMD[Ncut],no_error=True)
logger.info()
logger.info('gammaBMA=',get_err_str(gamBMA,gamBMAe))
logger.info('q=',gaudif(gamBMA,gamBMAe,univ.gamma,1e-12))


L, TcL, TcLerr, chiL, chiLerr, MTcL, MTcLerr = readTable('fig2_BAL.d')
ydata, edata = error_prop(np.log,chiL,chiLerr)
data = clipRange(np.vstack((L,ydata,edata)),0,minVal=minL)
ydata, edata = data[1], data[2]
xdata = np.log(data[0])
fit = Fitter(linearFit,xdata,ydata,edata)
res, res_err, chidof = fit.try_fit(algorithms=['curve_fit'],start_params=[1,univ.gamma/univ.nu],show_results=True)
fit.plot_fit(color='purple',alpha=0.2)
fit.plot_data(color='purple',marker='s',label='BAL')
chidof = "{:.3f}".format(chidof)
set_params(xlabel='${\\rm log}~L$',ylabel='${\\rm log}~\\chi_{\\rm max}$')


# Try a BMA here
ICs, gams, gamerrs = [], [], []
for Ncut in [0,1,2,3,4]:#,5]:
    xcut   = xdata[Ncut:]
    ycut   = ydata[Ncut:]
    ecut   = edata[Ncut:]
    fitcut = Fitter(linearFit,xcut,ycut,ecut)
    rescut, reserrcut, _ = fitcut.try_fit(algorithms=['curve_fit'],start_params=[1,univ.gamma/univ.nu])
    IC = BAIC(xdata=xcut, ydata=ycut, cov=np.diag(ecut), func=linearFit, params=rescut, Ncut=Ncut)
    ICs.append(IC)
    gams.append(rescut[1])
    gamerrs.append(reserrcut[1])
ICs, gams, gamerrs = toNumpy(ICs, gams, gamerrs) 
gamBMA, gamBMAe = modelAverage(gams, gamerrs, ICs)
PrMD = getModelWeights(ICs)
for Ncut in [0,1,2,3,4]:#,5]:
    xcut   = xdata[Ncut:]
    ycut   = ydata[Ncut:]
    ecut   = edata[Ncut:]
    fitcut = Fitter(linearFit,xcut,ycut,ecut)
    rescut, reserrcut, _ = fitcut.try_fit(algorithms=['curve_fit'],start_params=[1,univ.gamma/univ.nu])
    fitcut.plot_fit(color='purple',linestyle='dashed',alpha=PrMD[Ncut],no_error=True)

logger.info()
logger.info('gammaBMA=',get_err_str(gamBMA,gamBMAe))
logger.info('q=',gaudif(gamBMA,gamBMAe,univ.gamma,1e-12))

saveFigure('gamma.pdf')
plt.show()
clearPlot()


finalize()
