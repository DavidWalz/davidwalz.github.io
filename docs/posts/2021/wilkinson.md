# Wilkinson formulas and `Formulaic`

Wilkinson formulas are a symbolic notation for expressing statistical models. They were in introduced in Wilkinson and Rogers 1973, Symbolic Description of Factorial Models for Analysis of Variance https://www.jstor.org/stable/2346786 and extended over time for mixed and nested models.
The notation is not standardized and there are differences in the individual implementations in R ([`formula`](https://cran.r-project.org/web/packages/Formula/)) and Matlab ([documentation](https://www.mathworks.com/help/stats/wilkinson-notation.html)) and Python ([`patsy`](https://github.com/pydata/patsy)) among others.

My observation is that Wilkinson notation is much less common in the Python stats community than in R. 
On reason may be that the development of `patsy` stopped in 2018 and the package started to collect open issues and deprecation warnings since then.
I remember looking into using it for my work, but decided against it for the lack of support.
Luckily there is now a designated successor called [`Formulaic`](https://github.com/matthewwardrop/formulaic).
While `Formulaic` is still in beta, it has almost reached feature parity with `patsy` and can be used as drop-in replacement in some use cases.
For example `statsmodels` is planning to [adopt Formulaic](https://github.com/statsmodels/statsmodels/issues/6858).


```python
import numpy as np
import pandas as pd
import formulaic
from formulaic import Formula

print(formulaic.__version__)
```

    0.2.4


## The notation

Wilkinson notation is best shown by example.
With `y ~ x1 + x2 + x3` we specify that `y` is modeled by a linear combination of the independent variables `x1`, `x2`, `x3`.
Note, that an intercept term is implicitly added.


```python
Formula("y ~ x1 + x2 + x3")
```




    y ~ 1 + x1 + x2 + x3



We can disable the intercept (e.g. if `y` is centered) by adding `-1`.


```python
Formula("y ~ x1 + x2 + x3 - 1")
```




    y ~ x1 + x2 + x3



Interactions between variables are denoted with the colon operator, e.g. `x1:x2` means that a $x_1 \cdot x_2$ term is added to the model.
Adding all pairwise interactions can be done by writing it out `x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3` or in a more convenient way using `(x1 + x2 + x3)**2`, where the `**2` is meant as 2-way interactions, not as a square.


```python
Formula("y ~(x1 + x2 + x3)**2")
```




    y ~ 1 + x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3



Similarly, three-way interactions can be added with `(x1 + x2 + x3)**3` and individual terms disabled with, e.g.  `- x1:x2` 


```python
Formula("y ~(x1 + x2 + x3)**3 - x1:x2")
```




    y ~ 1 + x1 + x2 + x3 + x1:x3 + x2:x3 + x1:x2:x3



Powers and other transformations can be added inside an `I(...)`. 
In Formulaic this can also be done with curly braces, e.g. `{x1**2}`.


```python
Formula("y ~ (x1 + x2 + x3)**2 + {x1**2} + {x2**2} + {x3**2}")
```




    y ~ 1 + x1 + x1**2 + x2 + x2**2 + x3 + x3**2 + x1:x2 + x1:x3 + x2:x3



Nested effects can be specified with `a/b` which adds a term for `a` and all interactions of `a` with `b`.
For example `x1 / (x2 + x3)` crosses `x1` with each of `x2`, `x3`:


```python
Formula("y ~ x1 / (x2 + x3)")
```




    y ~ 1 + x1 + x1:x2 + x1:x3



In case variables contain special characters, they need to be quoted in backticks.


```python
Formula("y ~ `x 1` + `x§` + `x#`")
```




    y ~ 1 + x 1 + x# + x§



The full grammar implemented in Formulaic is listed [here](https://matthewwardrop.github.io/formulaic/basic/grammar).
Mixed effects modelling using the `|` and `||` is not yet implemented.

## What's it for?
By passing a dataframe to the formula we get the design matrix `X`. 
Here it gets really interesting as categorical variables are automatically dummy-encoded and the formula expanded correspondingly.
In the dummy-encoding the first level ("A" in the example below) is always dropped.


```python
df = pd.DataFrame({
    "y": [0, 1, 2],
    "x1": [6, 2, 5],
    "x2": [0.3, 0.1, 0.2],
    "x3": ["A", "B", "C"]
})
y, X = Formula("y ~ (x1 + x2 + x3)**2").get_model_matrix(df)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intercept</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3[T.B]</th>
      <th>x3[T.C]</th>
      <th>x1:x2</th>
      <th>x3[T.B]:x1</th>
      <th>x3[T.C]:x1</th>
      <th>x3[T.B]:x2</th>
      <th>x3[T.C]:x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6</td>
      <td>0.3</td>
      <td>0</td>
      <td>0</td>
      <td>1.8</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2</td>
      <td>0.1</td>
      <td>1</td>
      <td>0</td>
      <td>0.2</td>
      <td>2</td>
      <td>0</td>
      <td>0.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>5</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



`X` and `y` are of type `ModelMatrix` but can be converted back to `pd.DataFrames`.


```python
type(X)
```




    formulaic.model_matrix.ModelMatrix



We can retreive the `model_spec` to expand other dataframes to the same design matrix.
This is useful when the new dataframe we pass does not contain all categorical levels.


```python
df2 = pd.DataFrame({
    "x1": [4, 3, 5],
    "x2": [0.4, 0.3, 0.1],
    "x3": ["B", "B", "A"]
})

spec = X.model_spec
spec.get_model_matrix(df2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intercept</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3[T.B]</th>
      <th>x3[T.C]</th>
      <th>x1:x2</th>
      <th>x3[T.B]:x1</th>
      <th>x3[T.C]:x1</th>
      <th>x3[T.B]:x2</th>
      <th>x3[T.C]:x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4</td>
      <td>0.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.6</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>3</td>
      <td>0.3</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.9</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>5</td>
      <td>0.1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## An example

To get a feeling for formulaic in practice, we'll model the BaumgartnerAniline dataset from [mopti](https://github.com/basf/mopti).


```python
import opti

problem = opti.problems.BaumgartnerAniline()
df = problem.data
problem
```




    Problem(
    name=Baumgartner 2019 - Aniline Cross-Coupling,
    inputs=Parameters(
    [Categorical(name='catalyst', domain=['tBuXPhos', 'tBuBrettPhos', 'AlPhos']),
     Categorical(name='base', domain=['TEA', 'TMG', 'BTMG', 'DBU']),
     Continuous(name='base_equivalents', domain=[1.0, 2.5]),
     Continuous(name='temperature', domain=[30, 100]),
     Continuous(name='residence_time', domain=[60, 1800])]
    ),
    outputs=Parameters(
    [Continuous(name='yield', domain=[0, 1])]
    ),
    objectives=Objectives(
    [Maximize('yield', target=0)]
    ),
    data=
       catalyst  base  base_equivalents  temperature  residence_time     yield
    0  tBuXPhos   DBU          2.183015         30.0      328.717802  0.042833
    1  tBuXPhos  BTMG          2.190882        100.0       73.331194  0.959690
    2  tBuXPhos   TMG          1.093138         47.5       75.121297  0.031579
    3  tBuXPhos   TMG          2.186276        100.0      673.259508  0.766768
    4  tBuXPhos   TEA          1.108767         30.0      107.541151  0.072299
    )



Now we're ready to try out a number of models.
We'll use a linear regressor from scikit-learn and evaluate using leave-one-out cross-validation $R^2$ score ($Q^2$).
Since the regressor includes an intercept term by default, we'll remove the one in the design matrix.
A linear model gives $Q^2 = 0.73$ and $R^2 = 0.78$ indicating non-linearities in the response.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score

formula = Formula("yield ~ catalyst + base + base_equivalents + temperature + residence_time - 1")
y, X = formula.get_model_matrix(df)
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))
q2 = r2_score(y, cross_val_predict(model, X, y, cv=LeaveOneOut()))
print(f"{formula}\nR²={r2:.2f} Q²={q2:.2f}")
```

    yield ~ base + base_equivalents + catalyst + residence_time + temperature
    R²=0.78 Q²=0.73


With all interactions terms this increases to $Q^2 = 0.87$ and $R^2 = 0.95$.


```python
formula = Formula("yield ~ (catalyst + base + base_equivalents + temperature + residence_time)**2 - 1")
y, X = formula.get_model_matrix(df)
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))
q2 = r2_score(y, cross_val_predict(model, X, y, cv=LeaveOneOut()))
print(f"{formula}\nR²={r2:.2f} Q²={q2:.2f}")
```

    yield ~ base + base_equivalents + catalyst + residence_time + temperature + base:base_equivalents + base:catalyst + base:residence_time + base:temperature + base_equivalents:catalyst + base_equivalents:residence_time + base_equivalents:temperature + catalyst:residence_time + catalyst:temperature + residence_time:temperature
    R²=0.95 Q²=0.87


Adding 3-way interactions does not improve the fit.


```python
formula = Formula("yield ~ (catalyst + base + base_equivalents + temperature + residence_time)**3 - 1")
y, X = formula.get_model_matrix(df)
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))
q2 = r2_score(y, cross_val_predict(model, X, y, cv=LeaveOneOut()))
print(f"{formula}\nR²={r2:.2f} Q²={q2:.2f}")
```

    yield ~ base + base_equivalents + catalyst + residence_time + temperature + base:base_equivalents + base:catalyst + base:residence_time + base:temperature + base_equivalents:catalyst + base_equivalents:residence_time + base_equivalents:temperature + catalyst:residence_time + catalyst:temperature + residence_time:temperature + base:base_equivalents:catalyst + base:base_equivalents:residence_time + base:base_equivalents:temperature + base:catalyst:residence_time + base:catalyst:temperature + base:residence_time:temperature + base_equivalents:catalyst:residence_time + base_equivalents:catalyst:temperature + base_equivalents:residence_time:temperature + catalyst:residence_time:temperature
    R²=0.99 Q²=0.41


Sure, this can also be done with one-hot encoding and polynomial feature expansion in scikit-learn.
However, Formulaic is more expressive and prevents making errors such as not dropping a level in the one-hot encoding, or taking the square terms of dummy variables.

For comparison this is what it looks in statsmodels.
Note that we have to rename the columns as `patsy` doesn't like their names.


```python
from statsmodels.formula.api import ols

df2 = df.copy()
df2.columns = ["x1", "x2", "x3", "x4", "x5", "y"]
model = ols("y ~ (x1 + x2 + x3 + x4 + x5)**2", data=df2).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.951</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.927</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   38.44</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 07 Nov 2021</td> <th>  Prob (F-statistic):</th> <td>1.38e-30</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:30:35</td>     <th>  Log-Likelihood:    </th> <td>  92.203</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    96</td>      <th>  AIC:               </th> <td>  -118.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    63</td>      <th>  BIC:               </th> <td>  -33.78</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    32</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
                <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                    <td>    1.2168</td> <td>    0.187</td> <td>    6.496</td> <td> 0.000</td> <td>    0.842</td> <td>    1.591</td>
</tr>
<tr>
  <th>x1[T.tBuBrettPhos]</th>           <td>   -0.1101</td> <td>    0.150</td> <td>   -0.733</td> <td> 0.466</td> <td>   -0.410</td> <td>    0.190</td>
</tr>
<tr>
  <th>x1[T.tBuXPhos]</th>               <td>   -0.3721</td> <td>    0.152</td> <td>   -2.444</td> <td> 0.017</td> <td>   -0.676</td> <td>   -0.068</td>
</tr>
<tr>
  <th>x2[T.DBU]</th>                    <td>   -0.5980</td> <td>    0.172</td> <td>   -3.478</td> <td> 0.001</td> <td>   -0.941</td> <td>   -0.254</td>
</tr>
<tr>
  <th>x2[T.TEA]</th>                    <td>   -0.6998</td> <td>    0.192</td> <td>   -3.636</td> <td> 0.001</td> <td>   -1.084</td> <td>   -0.315</td>
</tr>
<tr>
  <th>x2[T.TMG]</th>                    <td>   -1.3760</td> <td>    0.163</td> <td>   -8.432</td> <td> 0.000</td> <td>   -1.702</td> <td>   -1.050</td>
</tr>
<tr>
  <th>x1[T.tBuBrettPhos]:x2[T.DBU]</th> <td>   -0.2011</td> <td>    0.093</td> <td>   -2.159</td> <td> 0.035</td> <td>   -0.387</td> <td>   -0.015</td>
</tr>
<tr>
  <th>x1[T.tBuXPhos]:x2[T.DBU]</th>     <td>   -0.1683</td> <td>    0.087</td> <td>   -1.943</td> <td> 0.056</td> <td>   -0.341</td> <td>    0.005</td>
</tr>
<tr>
  <th>x1[T.tBuBrettPhos]:x2[T.TEA]</th> <td>   -0.1892</td> <td>    0.101</td> <td>   -1.868</td> <td> 0.066</td> <td>   -0.392</td> <td>    0.013</td>
</tr>
<tr>
  <th>x1[T.tBuXPhos]:x2[T.TEA]</th>     <td>   -0.0336</td> <td>    0.098</td> <td>   -0.344</td> <td> 0.732</td> <td>   -0.229</td> <td>    0.162</td>
</tr>
<tr>
  <th>x1[T.tBuBrettPhos]:x2[T.TMG]</th> <td>   -0.0267</td> <td>    0.087</td> <td>   -0.308</td> <td> 0.759</td> <td>   -0.200</td> <td>    0.147</td>
</tr>
<tr>
  <th>x1[T.tBuXPhos]:x2[T.TMG]</th>     <td>   -0.1010</td> <td>    0.083</td> <td>   -1.224</td> <td> 0.226</td> <td>   -0.266</td> <td>    0.064</td>
</tr>
<tr>
  <th>x3</th>                           <td>   -0.1204</td> <td>    0.096</td> <td>   -1.259</td> <td> 0.213</td> <td>   -0.311</td> <td>    0.071</td>
</tr>
<tr>
  <th>x1[T.tBuBrettPhos]:x3</th>        <td>   -0.0198</td> <td>    0.066</td> <td>   -0.301</td> <td> 0.765</td> <td>   -0.151</td> <td>    0.112</td>
</tr>
<tr>
  <th>x1[T.tBuXPhos]:x3</th>            <td>    0.0207</td> <td>    0.065</td> <td>    0.318</td> <td> 0.752</td> <td>   -0.109</td> <td>    0.151</td>
</tr>
<tr>
  <th>x2[T.DBU]:x3</th>                 <td>   -0.0130</td> <td>    0.071</td> <td>   -0.183</td> <td> 0.855</td> <td>   -0.155</td> <td>    0.129</td>
</tr>
<tr>
  <th>x2[T.TEA]:x3</th>                 <td>    0.0077</td> <td>    0.074</td> <td>    0.105</td> <td> 0.917</td> <td>   -0.140</td> <td>    0.155</td>
</tr>
<tr>
  <th>x2[T.TMG]:x3</th>                 <td>   -0.0497</td> <td>    0.074</td> <td>   -0.668</td> <td> 0.507</td> <td>   -0.198</td> <td>    0.099</td>
</tr>
<tr>
  <th>x4</th>                           <td>   -0.0056</td> <td>    0.002</td> <td>   -2.694</td> <td> 0.009</td> <td>   -0.010</td> <td>   -0.001</td>
</tr>
<tr>
  <th>x1[T.tBuBrettPhos]:x4</th>        <td>    0.0037</td> <td>    0.001</td> <td>    2.999</td> <td> 0.004</td> <td>    0.001</td> <td>    0.006</td>
</tr>
<tr>
  <th>x1[T.tBuXPhos]:x4</th>            <td>    0.0042</td> <td>    0.001</td> <td>    3.520</td> <td> 0.001</td> <td>    0.002</td> <td>    0.007</td>
</tr>
<tr>
  <th>x2[T.DBU]:x4</th>                 <td>    0.0077</td> <td>    0.001</td> <td>    6.078</td> <td> 0.000</td> <td>    0.005</td> <td>    0.010</td>
</tr>
<tr>
  <th>x2[T.TEA]:x4</th>                 <td>   -0.0005</td> <td>    0.001</td> <td>   -0.346</td> <td> 0.730</td> <td>   -0.003</td> <td>    0.002</td>
</tr>
<tr>
  <th>x2[T.TMG]:x4</th>                 <td>    0.0123</td> <td>    0.001</td> <td>   10.457</td> <td> 0.000</td> <td>    0.010</td> <td>    0.015</td>
</tr>
<tr>
  <th>x5</th>                           <td>    0.0002</td> <td>    0.000</td> <td>    1.414</td> <td> 0.162</td> <td>-7.52e-05</td> <td>    0.000</td>
</tr>
<tr>
  <th>x1[T.tBuBrettPhos]:x5</th>        <td> -9.09e-05</td> <td> 7.14e-05</td> <td>   -1.273</td> <td> 0.208</td> <td>   -0.000</td> <td> 5.18e-05</td>
</tr>
<tr>
  <th>x1[T.tBuXPhos]:x5</th>            <td> 3.655e-06</td> <td> 6.63e-05</td> <td>    0.055</td> <td> 0.956</td> <td>   -0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>x2[T.DBU]:x5</th>                 <td> 1.023e-05</td> <td> 7.86e-05</td> <td>    0.130</td> <td> 0.897</td> <td>   -0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>x2[T.TEA]:x5</th>                 <td>-9.419e-05</td> <td> 7.73e-05</td> <td>   -1.219</td> <td> 0.227</td> <td>   -0.000</td> <td> 6.02e-05</td>
</tr>
<tr>
  <th>x2[T.TMG]:x5</th>                 <td>    0.0002</td> <td> 6.87e-05</td> <td>    2.739</td> <td> 0.008</td> <td> 5.09e-05</td> <td>    0.000</td>
</tr>
<tr>
  <th>x3:x4</th>                        <td>    0.0022</td> <td>    0.001</td> <td>    2.495</td> <td> 0.015</td> <td>    0.000</td> <td>    0.004</td>
</tr>
<tr>
  <th>x3:x5</th>                        <td>-3.465e-05</td> <td> 5.26e-05</td> <td>   -0.659</td> <td> 0.512</td> <td>   -0.000</td> <td> 7.04e-05</td>
</tr>
<tr>
  <th>x4:x5</th>                        <td> -1.93e-07</td> <td> 1.04e-06</td> <td>   -0.186</td> <td> 0.853</td> <td>-2.27e-06</td> <td> 1.89e-06</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>19.784</td> <th>  Durbin-Watson:     </th> <td>   1.865</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  34.781</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.832</td> <th>  Prob(JB):          </th> <td>2.80e-08</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.434</td> <th>  Cond. No.          </th> <td>1.83e+06</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.83e+06. This might indicate that there are<br/>strong multicollinearity or other numerical problems.


