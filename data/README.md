# Data files 

Here we collect data files needed to reproduce our figures.

Figure 1 can be created by running main_fig1.py. It simply
displays the results of average output layer for each temperature,
which is saved in the `avgP` subfolder.

To reproduce the data in Figure 2, one needs the needs the
energy per spin for each configuration to carry out the
reweighting. The energy densities are in the `MCMC` subfolder.
The to-be-reweighted observable here is the ML prediction.
The ML prediction for each configuration can be found
in the ML subfolder. Figure 2 can be created with main_fig2.py. 
The user specifies which subfigure to look at by changing 
`PLOTKIND` and `MLmodel` on lines 32-35 of that script.

The files `fig3_BAL.d` and `fig3_BAL20.d` contain the data
points used for the fits in Figure 3. The fit can be carried
out using `main_fig3.py`. An example log file showing the output
of this run is given in `example_fig3.log`.


## MCMC
Each subfolder gives the lattice extension.
Each file is labeled by the temperature.
The files are organized into three columns:
```
configuration    magnetization per spin    energy per spin
```

## ML 
Each subfolder gives the lattice extension and ML model.
Each file is labeled by the temperature.
The files are organized into two columns:
```
configuration    prediction 
```
