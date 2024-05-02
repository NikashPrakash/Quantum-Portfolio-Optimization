# Quantum Portfolio Optimization

## 1. Intro

**Solving the Markowitz's Portfolio Theory's optimization problem with Variational Quantum Eigensolvers (VQEs) and Quantum Error Mitigation (QEM).**
This project follows the approach in this paper: [Best Practices for PO by QC](https://www.nature.com/articles/s41598-023-45392-w).

The novel approach I am working to implement is a QEM method called Uniformly Decaying Subspaces (UDS). Benchmarking it by comparison against other QEM methods such as Zero-Noise Extrapolation, Probabilistic Error Cancellation, Measurement Error Mitigation

## 2. Methodology
### 2.1 Quadratic Formulation with Constriants
A portfolio is defined as the set of investments $x_i$ (measured as a fraction of the budget $B$) allocated for each *i*th asset of the market. 
The formulation of the Portfolio Optimization (PO) problem constrained under a budget is the quandratic objective function:
```math
\begin{equation}\begin{aligned}\underset{x}{\max }{\mathscr {L}}(x): \underset{x}{\max }(\mu ^{\text {T}} x - qx^{\text {T}}\Sigma x),\\ \text {s.t.} \quad \sum ^{N}_{i=1}x_i=1 \end{aligned}\end{equation}$$
```
Now considering $x$ is a possible solution to a problem with continous variables, the product $x_iB$ will be a multiple of $P_i$ - the closing price of the *i*th asset. To make to more computational feasable a subset of the problems where $x_iB$ is a integer multiple of $P_i$. A transformation of the variables $P,\ \mu,\ \Sigma,\text{ and } x$ will be necessary for the integer formulation:
```math
\begin{gathered} n = xB \\ P' = P/B \\ \mu'= P \circ \mu  && \text{where} \circ \text{is the Hadamard product} \\ \Sigma' = (P' \circ \Sigma)^T \circ P' \end{gathered}$$
```
```math
\begin{equation}\begin{gathered} \mathop {\max }\limits_{n} {\mathcal{L}}(n):\mathop {\max }\limits_{n} (\mu ^{{\prime {\text{T}}}} n - qn^{{\text{T}}} \Sigma ^{\prime } n), \\   {\text{s}}{\text{.t}}{\text{.}}\quad P^{{\prime {\text{T}}}} n = 1 \end{gathered}\end{equation}
```
*note: It is possible to extend this work to the case where the product $x_iB$ is continous, but the precision has to be limited. The approach will be detailed in [2.2.1](#221-converting-from-percent-to-binary)*

### 2.2 Converting Integer to Binary
To get the number of binary digits to have we first get the maximum number of integer units able to be bought within the budget each asset $n_i^{max}$ and the get the number of binary digits we take the log:
```math
\begin{equation}
\begin{gathered} 
    n^{max}_{i} = Int\left(\dfrac{B}{P_i}\right)  \\
    d_i = Int\left(\log_2{n_i^{max}}\right)
\end{gathered}
\end{equation}
```
We then get binarized target variables $b_i = [b_{1,0},\dots,b_{1,d_{1}},\dots,b_{N,0},\dots,b_{N,d_{N}}] \in R^{dim(b)}$, where $dim(b) = \sum\limits^N_{i=1}(d_i+1)$

Which helps define encoding matrix $C \in R^{N \times dim(b)}$, where N is the number of assets:
```math
C = \begin{pmatrix} 2^0 & \cdots & 2^{d_{1}} & 0 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\ 0 & \cdots & 0 & 2^0 & \cdots & 2^{d_{2}} & \cdots & 0 & \cdots & 0\\ \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots& \ddots & \vdots \\ 0 & \cdots & 0 & 0 & \cdots & 0 & \cdots & 2^0 & \cdots & 2^{d_{N}} \end{pmatrix}
```
Which is used to further transform variables to be compatible with binary variables:
```math
\begin{gathered} \mu'' = C^T\mu' \\ \Sigma'' = C^T\Sigma'C \\ P''=C^TP' \end{gathered}
```
This results in the Quadratic Binary Optimization Problem:
```math 
\begin{equation}\begin{gathered}   \mathop {\max }\limits_{b} {\mathcal{L}}(b):\mathop {\max }\limits_{b} \left( {\mu ^{{\prime \prime {\text{T}}}} b - qb^{{\text{T}}} \Sigma ^{{\prime \prime }} b} \right),  \\   {\text{s}}{\text{.t}}{\text{.}}\quad P^{{\prime \prime {\text{T}}}} b = 1 \\   \quad \quad b_{i}  \in \{ 0,1\} \quad \forall i \in \left[ {1, \ldots ,dim(b)} \right]. \end{gathered}\end{equation}
```
#### 2.2.1 Converting from Percent to Binary
My approach to converting a continous percentage of the budget $x_i$, if precision is limited two 2 decimal places then $x_i$ can be any integer from [0, 100]. Which would require 7 binary variables per asset. 
Ex:
- Given optimal percent of budget x_i for $i$th asset, 24%
- would be represented as 0011000 in the output string from measurement of the low-energy eigenstates from the Hamiltonian objective.


We then get binarized target variables $b_i = [b_{1,0},\dots,b_{1,d_{1}},\dots,b_{N,0},\dots,b_{N,d_{N}}] \in R^{dim(b)}$, where $dim(b) = 7*N$

Which helps define encoding matrix $C \in R^{N \times dim(b)}$, where N is the number of assets:
```math
C = \begin{pmatrix} 2^0 & \cdots & 2^{d_{1}} & 0 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\ 0 & \cdots & 0 & 2^0 & \cdots & 2^{d_{2}} & \cdots & 0 & \cdots & 0\\ \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots & \vdots& \ddots & \vdots \\ 0 & \cdots & 0 & 0 & \cdots & 0 & \cdots & 2^0 & \cdots & 2^{d_{N}} \end{pmatrix}
```

### 2.3 Quadratic Unconstrained Binary Optimization
To convert it to an unconstrained problem we transform the constrain into a penalty term in the objective function. Each constraint posed must have it's own penalty. The one in (4) becomes $\lambda(P''b-1)^2$:
```math
\begin{equation}\mathop {\max }\limits_{b} {\mathcal{L}}(b):\mathop {\max }\limits_{b} \left( {\mu ^{{\prime \prime {\text{T}}}} b - qb^{{\text{T}}} \Sigma ^{{\prime \prime }} b - \lambda (P^{{\prime \prime {\text{T}}}} b - 1)^{2} } \right).\end{equation}
```

### 2.4 Quantum Ising Hamiltonian 
To convert it into an Ising, related literature suggests expanding the components for ease of transformation:
```math
\begin{equation}\mathcal{L}(b):\sum\limits_{i} {\mu _{i}^{\prime } b_{i} }  - q\sum\limits_{{i,j}} {\Sigma _{{i,j}}^{\prime } } b_{i} b_{j}  - \lambda \left( {\sum\limits_{i} {P_{i}^{\prime } b_{i}  - 1} } \right)^{2}, \end{equation}
```
where $\mu'_i,\Sigma'_{i,j}, P'_i$ are components of $\mu_i'', \Sigma_i'', P''_i$, and $i \in [1,dim(b)].\\$ 
Now to convert into an Ising, spin variables $s_i$ which have values {-1,1}, are used in the transform $b_i \rightarrow \dfrac{1+s_i}{2}$. This results in a re-arrangement of the coefficients and gives:
```math
\begin{equation}\begin{gathered}\underset{s}{\min}\ {\mathscr {L}}(s): \underset{s}{\min }\left( \sum _{i}h_{i}s_{i}+ \sum _{i,j} J_{i,j}s_{i}s_{j}+\lambda (\sum _{i}(\pi _{i}s_{i}-\beta )^{2}\right)\\ \text {s.t.} \quad s_{i,j}\in \{-1,1\} \quad \forall i \nonumber,\end{gathered}\end{equation}
```
$J_{i,j}$ represents the coupling term between two spin variables.
We know that the eigenvalues of the Pauli Z operators are $\pm1$, which means it is suitable to represent the classical spin variables $s_i$. The two-body interaction can be represented through the tensor product of two Pauli Z Operators - $Z_i \otimes Z_{j}$..

The Qauntum Ising Hamiltonian:
```math
\begin{equation}\begin{gathered} \sum _{i}h_{i}Z_{i}+ \sum _{i,j} J_{i,j}Z_{i} \otimes Z_{j}+\lambda \sum _{i}(\pi _{i}Z_{i}-\beta )^{2}.\end{gathered}\end{equation}
```

### 2.5 VQE method

### 2.6 QAOA Method

### 2.7 UDS
