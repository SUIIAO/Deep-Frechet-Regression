# Deep Fréchet Regression

This repository contains codes necessary to replicate **Iao, Zhou and Müller (2025+)**: “Deep Fréchet Regression”. The `DFR` functions in the `code` folder, short for Deep Fréchet Regression, are designed for modeling the relationship between multivariate predictors and metric space-valued responses.

## Folder Structure

The folder structure of this repo is as follows:

| Folder      | Detail                                                                                                    |
|:------------|:----------------------------------------------------------------------------------------------------------|
| code        | R and Python scripts for the proposed approach                                                            |
| data        | Data files                                                                                                |
| simulation  | R scripts for simulations                                                                                 |

## code

| Data file.  | Detail                                                   |
|:------------|:---------------------------------------------------------|
| DFR.R       | Deep Fréchet Regression                                  |
| DNN.py      | Deep neural network in Python                            |
| NN_class.py | Class for deep neural network in Python                  |
| lrem.R      | Local Fréchet Regression for distributional data         |
| lnr.R       | Local Fréchet Regression for network data                |
| lcm.R       | Least common multiple for a vector of integers           |
| bwCV.R      | Bandwidth selection for local REM using cross-validation |
| kerFctn.R   | Kernel function                                          |
| le.R        | Laplacian eigenmaps function                             |
| dm.R        | Diffusion map function                                   |

## data

| Data file                | Detail                                         |
|:-------------------------|:-----------------------------------------------|
| taxi                     | New York taxi data application                 |
| mortaility               | Human mortality data applcaition               |

## simulation

R scripts to replicate simulation results in subsection 5.2 of the main text and Section S.2 of the Supplementary Material.

| Data file          | Detail                                                |
|:-------------------|:------------------------------------------------------|
| network            | Simulation for network data                           |
| distributional     | Simulation for distributional data                    |

## Report Errors

To report errors, please contact <siao@ucdavis.edu>. Comments and suggestions are welcome.

## Citation

"[Deep Fréchet regression](https://arxiv.org/pdf/2307.05726)"

```         
@article{iao2024deep,
  title={Deep Fr\'echet Regression},
  author={Iao, Su I and Zhou, Yidong and M{\"u}ller, Hans-Georg},
  journal={Journal of the American Statistical Association},
  volumn={just accepted},
  year={2024}
}
```
