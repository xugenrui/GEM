# GEM
The implementation of **G**enerative sky image prediction **E**nhanced **M**ultimodal fusion framework (**GEM**) for short-term solar irradiance forecasting.

We currently provide the implementation details of the core modules of the paper, namely the "Generative Sky Image Prediction enhanced Sky Image Encoder", the "Adaptive Fusion Module" and the "irradiance-guided cross-attention".

More implementation details will be updated since the publication.

## Code implementation details

All data (sky images and time series) is downloaded from [NREL Solar Radiation Research Laboratory](https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS#DOI) and [SIRTA Atmospheric Observatory](https://sirta.ipsl.polytechnique.fr/).

We have uploaded our implementation of the 10-minute alignment code for the SIRTA dataset. A more general version will be coming soon.

The sky image is first corrected by distortion in our implementation (this step can also be skipped according to existing studies).

The pretrained generative sky image prediction task is trained based on [OpenSTL](https://github.com/chengtan9907/OpenSTL).

## Acknowledgements

All code in this repository is based on the following repositories:

- Time-Series-Library https://github.com/thuml/Time-Series-Library
- OpenSTL https://github.com/chengtan9907/OpenSTL
- STWave https://github.com/LMissher/STWave
- ViT-pytorch https://github.com/lucidrains/vit-pytorch
