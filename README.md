# Cardiac-Digital-Twin-Purkinje
This repository contains the code developed for the study:
_Digital Twinning of the Human Ventricular Activation Sequence to Clinical 12-Lead ECGs and Magnetic Resonance Imaging Using Realistic Purkinje Networks for in Silico Clinical Trials_


Published in Medical Image Analysis: DOI

## Monodomain simulations

To numerically solve the monodomain equation we use an open-source high-performance GPU solver called <em>MonoAlg3D_C</em>, which is publicly available at the following repository: 

- [MonoAlg3D_C](https://github.com/rsachetto/MonoAlg3D_C)

### Reference:

Oliveira RS, Rocha BM, Burgarelli D, Meira Jr W, Constantinides C, dos Santos RW. <em>Performance evaluation of GPU parallelization, space‐time adaptive algorithms, and their combination for simulating cardiac electrophysiology.</em> Int J Numer Meth Biomed Engng. 2018;34:e2913. https://doi.org/10.1002/cnm.2913

## Purkinje network generation

To generate the extra branches for the biophysically-detailed Purkinje networks we use the same Purkinje generation method given in <em>Berg et al. (2023)</em>, which is also publicly available in an open-source repository:

- [Shocker](https://github.com/bergolho/Shocker)

### Reference:

Berg, L. A, Rocha, B. M., Oliveira, R. S., Sebastian, R., Rodriguez, B., de Queiroz, R. A. B., Cherry, E. M., dos Santos, R. W. <em>Enhanced optimization-based method for the generation of patient-specific models of Purkinje networks.</em> Nature Scientific Reports. https://www.doi.org/10.1038/s41598-023-38653-1
