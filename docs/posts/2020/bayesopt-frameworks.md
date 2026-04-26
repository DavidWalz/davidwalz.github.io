# Bayesian optimization frameworks

Bayesian optimization (BO) is a method for optimizing expensive black-box functions by means of a probabilistic surrogate model and an acquisition function that specifies a trade-off between exploration (improving the surrogate) and exploitation (finding the minimizer of the surrogate and thus of the black-box function).

As **surrogate model** typically Gaussians processes (GP) or tree-based models (RF, GBT ...) are used.
These have their own parameters that need to be estimated via maximum likelihood (ML) or integrated out.
The ML point estimate is typically searched for using a gradient-based local optimizer together with a global optimization heuristic such as multiple restarts to address the non-convexiy of the problem.
For integrating out the model parameters either Markov-chain Monte Carlo (MCMC) or approximate variational inference (VI) methods are used, together with an assumption on the prior distribution.

**Search spaces** define the possible inputs.
When applying BO for tuning hyperparameters of machine learning models the inputs include continuous, discrete and categorical variables and may include conditionals (think of having to choose beta1 and beta2 when selecting Adam).
When applying BO to optimization of real-world processes there are typically some (in-)equality constraints that need to be satisfied.

**Acquistion functions** either have a analytic form, or need to be approximated via Monte Carlo methods.
We want to find the minimizer of the acquistion over the search space in order to evaluate it next.
This is typically done using a gradient-free or gradient-based local optimizer or via MC-sampling, depending on the nature of the acquistion function.
The search space including its constraints needs to be handled by the optimizer.
In multi-objective BO the black-box function has multiple outputs. 
Here the acquisition function needs to guide the search towards exploring the Pareto front. 
An comparison of acquisition functions is given here: [single-objective](https://davidwalz.github.io/blog/bayesian%20optimization/2020/06/20/bayesopt-acquisitions-single.html), [multi-objective](https://davidwalz.github.io/blog/bayesian%20optimization/2020/06/20/bayesopt-acquisitions-multi.html).

There are a number of general-purpose (not focussing only on hyperparameter tuning) BO frameworks available in Python and R. 
In this post the main frameworks are compared in terms of supported features and development & support activity.

## Feature comparison

All packages in this comparison support GP surrogates with analytic acquisitions EI/PI/CB over continuous search spaces.
Beyond that it's interesting to note that no package currently provides all options in terms of surrogates, acquistions and search space, so it really depends on the type of problem you want to apply BO on.  

| Name | Surrogates | Hyperparameter handling | Single-objective acquisitions | Multi-objective acquisitions | Search space |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------- | ---------------------------- | ----------------------------------------------- | --------------- |
| [mlrMBO](https://mlr-org.github.io/mlrMBO) | GP, RF ([mlr](https://mlr.mlr-org.com/)) | ML | EI, CB, AEI, EQI, AdaCB | DIB | continuous, integer, categorical + constraints |
| [pyGPGO](http://pygpgo.readthedocs.io) | GP (native), GBT, RF, ET ([scikit-learn](https://scikit-learn.org)) | ML, MCMC ([pymc3](https://docs.pymc.io/)) | EI, PI, CB | | continuous, integer |
| [acikit-optimize](https://scikit-optimize.github.io) | GP, RF, GBT ([scikit-learn](https://scikit-learn.org)) | ML | EI, PI, CB | - | continuous, discrete, categorical + constraints | 
| [GPyOpt](https://github.com/SheffieldML/GPyOpt) | GP ([GPy](http://sheffieldml.github.io/GPy)) | ML, MCMC | EI, PI, CB | - | continuous, discrete, categorical + constraints | 
| [GPflowOpt](https://github.com/GPflow/GPflowOpt) | GP ([GPflow](https://github.com/GPflow/GPflow)) | ML | EI, PI, CB, MES, PF | HVPI | continuous |
| [BoTorch](https://botorch.org/) | GP ([GPyTorch](https://github.com/cornellius-gp/gpytorch)), extensible | ML, MCMC (Pyro)| EI, PI, CB, qMES, qKG, extensible | custom scalarizations, HVEI | continous + linear constraints |
| [Emukit](https://github.com/amzn/emukit) | GP ([GPy](http://sheffieldml.github.io/GPy)), extensible | ML | EI, PI, CB, ES, MES, PF | - | continuous, integer, categorical + constraints |
| [DragonFly](https://github.com/dragonfly/dragonfly) | GP (native) | ML, posterior sampling | EI, CB, PI, TTEI, TS | scalarization with CB/TS | continuous, categorical |

## Activity comparison

The following table gives an impression on the popularity and development activity of the frameworks, based on github statistics. 
The top-3 in terms of stars, contributors, commits and issues are scikit-optimize, botorch (together with Ax which builds on botorch) and mlrMBO.


```python
import util

df = util.compare_repos([
    'mlr-org/mlrMBO',
    'SheffieldML/GPyOpt',
    'scikit-optimize/scikit-optimize',
    'GPflow/GPflowOpt',
    'pytorch/botorch',
    'amzn/emukit',
    'dragonfly/dragonfly',
    'josejimenezluna/pyGPGO',
    'facebook/Ax'
])
df.sort_values(by='closed_issues', ascending=False)
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
      <th>stars</th>
      <th>forks</th>
      <th>contributors</th>
      <th>commits</th>
      <th>open_issues</th>
      <th>closed_issues</th>
      <th>created</th>
      <th>last_commit</th>
      <th>license</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>scikit-optimize</th>
      <td>1836</td>
      <td>349</td>
      <td>55</td>
      <td>1500</td>
      <td>150</td>
      <td>769</td>
      <td>2016-03-20</td>
      <td>2020-05-18</td>
      <td>BSD-3-Clause</td>
    </tr>
    <tr>
      <th>botorch</th>
      <td>1630</td>
      <td>137</td>
      <td>28</td>
      <td>656</td>
      <td>24</td>
      <td>437</td>
      <td>2018-07-30</td>
      <td>2020-06-24</td>
      <td>MIT</td>
    </tr>
    <tr>
      <th>mlrMBO</th>
      <td>162</td>
      <td>40</td>
      <td>14</td>
      <td>1600</td>
      <td>84</td>
      <td>410</td>
      <td>2013-10-23</td>
      <td>2020-06-15</td>
      <td>NOASSERTION</td>
    </tr>
    <tr>
      <th>Ax</th>
      <td>1179</td>
      <td>113</td>
      <td>46</td>
      <td>608</td>
      <td>21</td>
      <td>318</td>
      <td>2019-02-09</td>
      <td>2020-06-29</td>
      <td>MIT</td>
    </tr>
    <tr>
      <th>emukit</th>
      <td>218</td>
      <td>64</td>
      <td>20</td>
      <td>282</td>
      <td>31</td>
      <td>278</td>
      <td>2018-09-04</td>
      <td>2020-06-03</td>
      <td>Apache-2.0</td>
    </tr>
    <tr>
      <th>GPyOpt</th>
      <td>621</td>
      <td>190</td>
      <td>35</td>
      <td>504</td>
      <td>89</td>
      <td>235</td>
      <td>2014-08-13</td>
      <td>2020-03-19</td>
      <td>BSD-3-Clause</td>
    </tr>
    <tr>
      <th>GPflowOpt</th>
      <td>213</td>
      <td>51</td>
      <td>5</td>
      <td>426</td>
      <td>22</td>
      <td>97</td>
      <td>2017-04-28</td>
      <td>2018-09-12</td>
      <td>Apache-2.0</td>
    </tr>
    <tr>
      <th>dragonfly</th>
      <td>498</td>
      <td>57</td>
      <td>8</td>
      <td>393</td>
      <td>17</td>
      <td>49</td>
      <td>2018-04-20</td>
      <td>2020-03-13</td>
      <td>MIT</td>
    </tr>
    <tr>
      <th>pyGPGO</th>
      <td>182</td>
      <td>47</td>
      <td>2</td>
      <td>292</td>
      <td>7</td>
      <td>18</td>
      <td>2016-11-23</td>
      <td>2019-06-15</td>
      <td>MIT</td>
    </tr>
  </tbody>
</table>
</div>



## GPyOpt
* GPy [[code]](https://github.com/SheffieldML/GPy) [[doc]](https://gpy.readthedocs.io/en/latest/)
* GPyOpt [[code]](https://github.com/SheffieldML/GPyOpt) [[doc]](https://gpyopt.readthedocs.io/en/latest/)

GPyOpt is a BO package built on top of the hugely popular GPy for flexible GP modeling. Both are being developed by the university of Sheffield.
Together with mlrMBO, GPyOpt has been around the longest. However, development of GPyOpt has somewhat stalled.

## Scikit-Optimize
[[code]](https://github.com/scikit-optimize/scikit-optimize)
[[doc]](https://scikit-optimize.github.io/)

Scikit-Optimize is an actively developed and well polished BO package based on the GP and tree-based models in Scikit-Learn. Not supported are model parameter integration and multi-objective optimization. Compared to GPy, the GP modeling in Scikit-Learn is rather rudimentary. Mixed search spaces and constraints are supported, as well as external, delayed and batched evaluations. For Hyperparameter tuning in Scikit-Learn there is a drop-in replacement for Grid/RandomSearchCV.

## GPFlowOpt (with TF & GPFlow)
* GPFlow [[code]](https://github.com/GPflow/GPflow) [[doc]](https://gpflow.readthedocs.io/en/latest/) [[paper]](https://arxiv.org/abs/1711.03845)
* GPFlowOpt [[code]](https://github.com/GPflow/GPflowOpt) [[doc]](https://gpflowopt.readthedocs.io/en/latest/) [[paper]](http://jmlr.org/papers/v18/16-537.html)

GPFlowOpt is package built on top of GPFlow, which in turn uses TensorFlow for fast linear algebra computations with GPU-support and auto-differentiation. This makes it more extensible as different models and acquisition functions can be implemented without having to define gradients for the optimizer.
The top-level API is inspired GPy.

*Development on GPFlowOpt seems to have stopped since the end of 2018.*

## BoTorch (with PyTorch, GPyTorch & Pyro)
* BoTorch [[code]](https://github.com/pytorch/botorch) [[doc]](https://botorch.org/) [[paper]](https://arxiv.org/abs/1910.06403)
* GPyTorch [[code]](https://github.com/cornellius-gp/gpytorch) [[doc]](https://gpytorch.ai/) [[paper]](https://arxiv.org/abs/1809.11165)
* Pyro [[code]](https://github.com/pyro-ppl/pyro) [[doc]](http://docs.pyro.ai) [[paper]](https://arxiv.org/abs/1810.09538)

BoTorch is a package built on top of GPyTorch (GP modeling), Pyro (MCMC and variational inference) and PyTorch as its compution framework. Hence, it profits from GPU support and auto-differentiation. BoTorch is extremely flexible, e.g. in it's first class support for custom acquisition functions, which is made possible by MC integration (quasi-MC together with the reparametrisation trick) and auto-differentiation for gradient-based optimization. BoTorch supports:
* GP models (multi-fidelity, multi-task, ...), variational neural networks 
* MC handling of model parameters is in principle supported via GPyTorch and Pyro, but is not yet implemented
* analytic acquisitions (EI, PI, CB) and MC acquisitions: (knowledge gradient, max-value entropy search, posterior variance), cost awareness and custom acquisitions
* multi-objective optimization via passing a custom pytorch function that scalarizes the objectives to the corresponding MC acqusisition
* only continous search spaces; categorical / ordinal variables need to be encoded beforehand
* parameter constraints (linear inequality constraints) and outcome constraints
* batched proposals

## Ax
[[code]](https://github.com/facebook/Ax)
[[doc]](https://ax.dev)
[[test]](ax.md)

Ax is a high-level framework for Bayesian and Bandit optimization. 
For BO, Ax relies mostly on BoTorch from the same developers, hence the entire feature set of BoTorch is available in principle, but needs to implemented first.
For Bandits optimization, Thompson sampling is used.
Ax provides:
* a service-like API for managing and storing experiments as JSON files (local) or in SQL databases. The service is only running locally.
* transform pipelines to handle encoding of categorical and ordinal variables, scaling and log-transforms
* limited support for parameter constraints of type $x1 \leq x2$ and $\sum x_i \leq c$ as well as output constraints
* limited support for multi-objective optimization: only weighted sum scalarization and no tooling around it
* examples for human-in-the-loop optimization

Overall, I think Ax needs more maturing.

## Emukit
[[code]](https://github.com/amzn/emukit)
[[doc]](https://amzn.github.io/emukit/)
[[paper]]()

Emukit is a high-level framework for Bayesian optimization and and Bayesian quadrature. It is intended to be independent of the modeling framework, but supports first class support for GPy. Similar built-in support for other frameworks is apparently not planned. Mixed search spaces and constraints are supported, multi-objective optimization is currently not.

Emukit provides abstraction layers for the individual components of Bayesian optimization in order to implement algorithms independent of the concrete modeling framework. While the idea is intriguing, concepts like auto-differentiation that make GPFlowOpt and GPyTorchOpt powerfull fall short here. Instead, by focussing on GPy, Emukit seems to end up as replacement for GPyOpt.

## Dragonfly
[[code]](https://github.com/dragonfly/dragonfly)
[[doc]](https://dragonfly-opt.readthedocs.io)
[[paper]](https://arxiv.org/abs/1903.06694)

Dragonfly is package developed at Carneggie Melon University. It has native implementations of GPs with the typical kernels, an optimizer (DOO), MCMC samplers copied from copied from pymc3 and pgmpy (Metropolis, Slice, NUTS, HMC). Multi-objective optimization is supported via random scalarizations, constraints are not.
Note that Thompson sampling from GPs seems to be incorrectly implemented, as points are sampled without keeping track of the previously sampled points.

## Other projects
* [fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization) - Small BO package using GPs from Scikit-Learn 
* [Cornell-MOE](https://github.com/wujian16/Cornell-MOE) - Relatively old Python / C++ package for BO
* [ProcessOptimizer](https://github.com/bytesandbrains/ProcessOptimizer) - Fork of Scikit-Optimizer for optimizing real world processes. Provides classes for composable constraints. However, only rejection sampling is implemented.
* [Phoenics](https://github.com/aspuru-guzik-group/phoenics) - BO package using kernel density estimates and a specific multi-objective acquistion called CHIMERA
* [TS-EMO](https://github.com/Eric-Bradford/TS-EMO) - Matlab implementation of the TS-EMO algorithm



```python

```
