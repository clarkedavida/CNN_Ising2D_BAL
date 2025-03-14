# Data files 

Here we collect data files needed to reproduce our figures.
To reproduce the data in Figure 1, one needs the needs the
energy per spin for each configuration to carry out the
reweighting. The necessary information is in the MCMC subfolder.

The to-be-reweighted observable here is the ML prediction.
The ML prediction for each configuration can be found
in the ML subfolder.

Figure 1 can be created with main_fig1.py. The user specifies which
subfigure to look at by changing `PLOTKIND` and `MLmodel` on lines
32-35 of that script.

The files `fig2_BAL.d` and `fig2_BAL20.d` contain the data
points used for the fits in Figure 2. The fit can be carried
out using `main_fig2.py`. An example log file showing the output
of this run is given in `example_fig2.log`.


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
