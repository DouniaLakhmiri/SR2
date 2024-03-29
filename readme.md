<h1 align="center">SR2: Training neural networks with sparsity inducing regularizations 
.</h1>

<p align="center">
    <a href="PyTorch">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"> </a>
    <a href="Python">
    <img src="https://img.shields.io/pypi/pyversions/gym_simplifiedtetris?style=for-the-badge"> </a>
    <a href="Licence">
    <img src="https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge"> </a>
    
</p>
      
<p align="center">
  <a href="#description">Description</a> •
  <a href="#train-a-network">Train a network</a> •
  <a href="#results">Results</a> •
  <a href="#references">References</a>
</p>

---


## Description
SR2 is an optimizer that trains deep neural networks with nonsmooth and non convex regularizations to retrieve a sparse and efficient sub-structure.

The optimizer minimizes a the sum of a finite-sum loss function $f$ and a nonsmooth nonconvex regularizer $\mathcal{R}$: 

$$ F(x) =f(x) + \lambda \mathcal{R}(x). $$
    
with an adaptive proximal quadratic regularization scheme.

Supported regularizers are $\ell_p^p$ with $p \in {0, \frac{1}{2}, \frac{2}{3}, 1}$:
- $||x||_0$ is the number of non zero $x_i$
- $||x||_p = (\sum_i |x_i|^p)^{\frac{1}{p}}$ for $p = \frac{1}{2}, \frac{2}{3}, 1$

---
        
## Train a network

### Prerequisits
 - Numpy
 - Pytorch
 - PyHessian [https://github.com/amirgholami/PyHessian]
 
### Run SR2

You can start training the network by running a command similar to

```
python main.py --reg=l0 --precond=andrei --beta=0.95 --lam=0.001
```

The following table gives a summary of the options and a brief description:

  SR2 option     | Description | Possible values |
| -------------  | ----------- | --------------- |
| --lam | $\lambda$ in $\lambda \mathcal{R}(x)$| $\mathbb{R}$| 
| --reg | Regularization function $\mathcal{R}(x)$| l0, l1, l12, l23 | 
| --beta | Momentum factor| $[0, 1]$ | 
| --precond | Choice of preconditioner to accelerate training| none, adam, andrei* | 
| --max_epoch | Number of training epochs| $\mathbb{N}$ | 
| --wd | Weight decay| $[0, 1]$ | 
| --seed | Random seed| $\mathbb{N}$ | 

---
        
## Results

--- 
        
## References 

```
@misc{https://doi.org/10.48550/arxiv.2206.06531,
  doi = {10.48550/ARXIV.2206.06531}, 
  url = {https://arxiv.org/abs/2206.06531},
  author = {Lakhmiri, Dounia and Orban, Dominique and Lodi, Andrea},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), Optimization and Control (math.OC), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Mathematics, FOS: Mathematics},
  title = {A Stochastic Proximal Method for Nonsmooth Regularized Finite Sum Optimization},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```
