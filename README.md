# A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining

**ICML 2023**

Shengchao Liu<sup>+</sup>, Weitao Du<sup>+</sup>, Zhiming Ma, Hongyu Guo, Jian Tang

<sup>+</sup> Equal contribution

[[Project Page](https://chao1224.github.io/MoleculeSDE)]
[[ArXiv]()]

<p align="center">
  <img src="figure/pipeline.png" /> 
</p>

- MoleculeSDE is GraphMVPv2, follow-up of GraphMVP
- It includes two components:
    - Contrastive learning
    - Generative learning:
        - One 2D->3D diffusion model. Frame-based SE(3)-equivariant and reflection anti-symmetric model
        - One 3D->2D diffusion model. SE(3)-invariant.

<p align="left">
  <img src="figure/demo.gif" width="100%" /> 
</p>

## Cite Us

Feel free to cite this work if you find it useful to you!

```
@inproceedings{liu2023moleculeSDE,
	title        = {A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining},
	author       = {Shengchao Liu and Weitao Du and Zhiming Ma and Hongyu Guo and Jian Tang},
	year         = 2023,
	booktitle    = {International Conference on Machine Learning},
}
```