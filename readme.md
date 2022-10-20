<h1 align="center">SR2: Training neural networks with sparsity inducing regularizations 
.</h1>

<p align="center">
    <a href="PyTorch">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
    <a href="Python">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <a href="Ubuntu">
    <img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white">
    <a href="macOS">
    <img src="https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0">
</p>
      
<p align="center">
  <a href="#description">Description</a> •
  <a href="#prerequisits">Prerequisits</a> •
  <a href="#results">Results</a> •
  <a href="#references">References</a>
</p>

---


## Description
SR2 is an optimizer that trains deep neural networks with nonsmooth and non convex regularizations to retrieve a sparse and efficient sub-structure.

The optimizer minimizes a the sum of a finite-sum loss function $f$ and a nonsmooth nonconvex regularizer $\mathcal{R}$: 

$$ F(x) =f(x) + \lambda \mathcal{R}(x). $$
    
with an adaptive proximal quadratic regularization scheme.

Supported regularizers are $\ell_0$ and $\ell_1$.

## Prerequisits 
 - Numpy
 - Pytorch
 - PyHessian [https://github.com/amirgholami/PyHessian]
 
 
## Results


## References 
