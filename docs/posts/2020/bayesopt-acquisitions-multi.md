# Multi-objective acquisition functions

Bayesian optimization is a sample-efficient method for solving $\min_{x \in \mathcal{X}} f(x)$ where $f$ is a black-box function whose evaluations are expensive and possibly noisy.

When $f$ is vector-valued, i.e. there are multiple outputs that we care to minimize, there is in general no single optimal point.
Instead we have a set of optimal compromises (the Pareto front), where for each point we cannot improve any one output without worsening at least one other output.

There are two approaches for dealing with multiple objectives in the context of Bayesian optimization: 
1) Use a multi-objective scalarization together with a single-objective acquisition 
2) Use a multi-objective acquistion that aims for improving the knowledge of the Pareto front.

Regarding 1. the same scalarizations as in multi-objective optimization may be applied. An overview of scalarization methods in the context of Bayesian optimization is given in [Chugh2019].
Regarding 2. there is a growing list of multi-objective acquistions

> Note: this post is work in progress.

## Expected hypervolume improvement
Expected hypervolume improvement (EHVI), also known as S-metric-based expected improvement, is an extension of the expected improvement to multiple objectives. 
It's the expectation value of the improvement in hypervolume for a given candidate $x$ with respect to the existing data $\mathcal{D}$.

The first method to calculate EVHI was Monte Carlo integration [Emmerich2006], however the accuracy of this method strongly depends on the number of MC samples.
For the 2D case an exact methods was proposed in [Emmerich2012] with complexity $O(n^3 \log n)$, where $n$ is the number of non-dominated points in the data set.
An exact method for >2D was proposed in [Cockuyt2014] without an analysis of the complexity.
[Hupkens2015] introduced a method with complexity of $O(n^2)$ for 2D and $O(n^3)$ for 3D.
Asymptotically optimal algorithms with $O(n \log n)$ complexity were proposed in [Emmerich2016] for the 2D case, and in [Yang2017] for the 3D case.
In [Yang2019] this was extended to >4D and to the probability of improvement on the Pareto front.
Codes are found [here](http://liacs.leidenuniv.nl/~csmoda/).

[Daulton2020] proposed q-EHVI (a quasi-MC formulation) with exact gradients via auto-differentiation to enable gradient based optimization. An implementation in BoTorch is anounced.

## Further links

* Svenson+ (2016) Multiobjective optimization of expensive-to-evaluate deterministic computer simulator models  
  https://www.sciencedirect.com/science/article/pii/S0167947315001991
* Keane (2012) Statistical Improvement Criteria for Use in Multiobjective Design Optimization  
  https://arc.aiaa.org/doi/abs/10.2514/1.16875?journalCode=aiaaj
* Garrido-Merchan+ (2019) Predictive Entropy Search for Multi-objective Bayesian Optimization with Constraints  
  https://www.sciencedirect.com/science/article/pii/S0925231219308525
* Picheny (2015) Multiobjective optimization using Gaussian process emulators via stepwise uncertainty reduction  
  https://link.springer.com/article/10.1007/s11222-014-9477-x
* Bradford (2018) Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm  
  https://doi.org/10.1007/s10898-018-0609-2
* Deutz (2019) Expected R2 Indicator Improvement as an Infill Criterion in Bi-objective Bayesian Optimization  
  https://link.springer.com/chapter/10.1007/978-3-030-12598-1_29
* Zachary+ (2019) Assessing the Frontier: Active Learning, Model Accuracy, and Multi-objective Materials Discovery and Optimization  
  https://arxiv.org/abs/1911.03224
