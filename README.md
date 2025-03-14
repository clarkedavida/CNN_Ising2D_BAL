# CNN_Ising2D_BAL

In [2501.05547](https://arxiv.org/abs/2501.05547)
we studied the effect of example size on the output layer of
a CNN trained to classify phases of the 2-$d$ Ising model.
Here we collect our CNN along with the data and scripts needed
to produce the plots in that paper. 

The CNN architecture is our best
understanding of the architecture presented in 
Appendix B of [2004.14341](https://arxiv.org/abs/2004.14341)
by [D. Bachtis](https://github.com/dbachtis),
G. Aarts, and B. Lucini.
This implementation by [A. Abuali](https://github.com/SabryPhys) utilizes 
[Scikit-Learn](https://github.com/scikit-learn/scikit-learn) and 
[Tensorflow](https://github.com/tensorflow/tensorflow) with 
[Keras](https://github.com/keras-team/keras).

To reproduce the figures, you need the 
[AnalysisToolbox](https://github.com/LatticeQCD/AnalysisToolbox). 
You can install the Toolbox using
```
pip install latqcdtools
```
Then navigate to the `data` subfolder and try using the scripts there.
`README.md` in that subfolder explains the data in more detail.
