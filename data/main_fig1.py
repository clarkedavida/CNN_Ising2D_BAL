# 
# main_fig1.py                                                               
# 
# D. Clarke
# 

from latqcdtools.base.plotting import plot_lines, getColorGradient, plt, set_params, set_default_param
from latqcdtools.base.readWrite import readTable
import latqcdtools.base.logger as logger


#MLmodel='BAL'
MLmodel='BAL20'

set_default_param(labelsintoplot=False,surroundWithTicks=False,font_size=12)
Lcolors=getColorGradient(9,'hsv')

if MLmodel=='BAL':
    Ls = [128, 200, 256, 360, 440, 512, 640, 760]
elif MLmodel=='BAL20':
    Ls = [128, 200, 256, 360, 440, 512, 640]
else:
    logger.TBerror("I don't recognize ML model",MLmodel)

iL=1
for L in Ls:
    T, P, Pe = readTable(f'ML/avgP/P{MLmodel}_{L}.d') 
    plot_lines(T,P,Pe,color=Lcolors[iL],label=f'{L}',marker=None)
    iL+=1

set_params(ylabel='$\\langle P\\rangle$',xlabel='$T$',title=MLmodel)
plt.show()
