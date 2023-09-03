The examples in this directory are based on the benchmark cases from Turner et
al. (Phys. Plasmas 20, 013507, 2013).
See the 'Examples' section in the documentation for previously computed
comparisons between WarpX and the literature results.
The 1D PICMI input file can be used to reproduce the results from Turner et al.
for a given case, N, by executing:
    `python3 PICMI_inputs_1d.py -n N`

The 1D PICMI input file also includes a test of the DSMC module in WarpX,
comparing to the same test case of Turner. The kinetic neutral population is
periodically reset to the starting temperature to mimic the effect of the non-
evolving background of the MCC case.
