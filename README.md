# GEM
The implementation of **G**enerative sky image prediction **E**nhanced **M**ultimodal fusion framework (**GEM**) for short-term solar irradiance forecasting

## Code implementation details
All data (sky images and time series) is downloaded from [NREL Solar Radiation Research Laboratory](https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS#DOI).
The sky image is first corrected by distortion (this step can also be skipped according to existing studies).
The pretrained generative sky image prediction task is trained based on [OpenSTL](https://github.com/chengtan9907/OpenSTL), which can be implemented by copying [config.py] [dataloader_srrl.py] [srrl.py] to run in OpenSTL;
Once you get the pretrained weights, you can proceed multimodal solar irradiance forecasting based on the code we provide.
The following is an example:
```
bash ./scripts/SRRL.conf 20241203 
```

## Acknowledgements
All code in this repository is based on the following repositories:
- Time-Series-Library [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- OpenSTL [https://github.com/chengtan9907/OpenSTL](https://github.com/chengtan9907/OpenSTL)
- STWave [https://github.com/LMissher/STWave](https://github.com/LMissher/STWave)
