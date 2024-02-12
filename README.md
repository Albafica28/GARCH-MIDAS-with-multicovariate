# Functions
## GARCHmidasFit.m
- This code is an implementation about GARCH-MIDAS model with multiple macroeconomic variables. More details can be found in Ghysels, E. (2016). MIDAS matlab toolbox.
- Input:
    - `X`: numeric vector, data of the high-frequency variable
    - `Y`: numeric matrix, data of multiple macroeconomic variables
    - Name-Value arguments:
        - `XDate`: date vector, the date of X
        - `YDat`e: date vector, the date of Y
        - `isGJR`: logical scalar, Whether or not the GARCH equation contains an asymmetric term
        - `nLags`: numeric integer, Lag order of low-frequency variables in the beta function
        - `isDisplay`: logical scalar, whether to print the details of the optimization process on the screen
        - `nPeriods`: numeric integer, number of the high-frequency variable included between neighboring macroeconomic variables
        - `mu`, `alphe` ...: numeric scalar, initial value of parameters
- Output:
    - `result`: struct, model fitting result
    - `result.resultTab`: table, parameter estimation result
    - `result.logLik`: numeric scalar, negative log likelihood
    - `result.AIC`: numeric scalar, AIC information value
    - `result.BIC`: numeric scalar, BIC information value
    - `sigmat`: numeric vector, conditional standard deviation
    - `zt`: numeric vector, innovation

## logLikelihood.m
- Some of the code references the MIDAS package by Hang Qian, see https://www.mathworks.com/matlabcentral/fileexchange/45150-midas-matlab-toolbox for more details.
- Try to enter the correct date `XDate` of `X` and `YDate` of `Y` for correct matching of low and high frequency data.

# Example
- see an example in `demo.m`.

# Dateset
- The high-frequency data are returns of `S&P500`, `U.S. 10-year Bond` and `Bitcoin` for the financial market and `S&P500 Green Bond Index` for the green bond market, respectively.
- The low-frequency data are `EPU` uncertainty index and `GPR` uncertainty index.
- see the dateset in `dataset.mat`.