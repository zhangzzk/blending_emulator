Overlapping galaxies (“blending”) induce **redshift mixing** in the effective source redshift distribution $n_{\gamma}(z)$, which can bias weak-lensing observables and cosmological inference.

This repository proposes an **efficient and flexible correction method** for blending-induced bias, described in detail in:

> Zhang, Z., Gruen, D., Tortorelli, L., Li, S.-S., & McCullough, J. (2025)  
> *Emulating redshift-mixing due to blending in weak gravitational lensing*  
> arXiv:2507.19130

Given a galaxy population model, `blending_predictor.ipynb` computes the **corrected effective redshift distribution** $n_{\gamma}(z)$ accounting for redshift mixing due to blending.

An example application to **HSC-like tomographic bins** is provided.
