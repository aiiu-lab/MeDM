# MeDM: Mediating Image Diffusion Models for Video-to-Video Translation with Temporal Correspondence Guidance

<img src="teaser.gif" width="100%"></img>

[Ernie Chu](https://ernestchu.github.io),
Tzuhsuan Huang,
Shuo-Yen Lin,
[Jun-Cheng Chen](https://www.citi.sinica.edu.tw/pages/pullpull/)

#### [Project Page](https://medm2023.github.io) | [Paper](https://medm2023.github.io/medm.pdf) | [arXiv](https://arxiv.org/abs/2308.10079) | [Colab](#colab-demo)

## Colab demo
- [Ground-truth flow](https://colab.research.google.com/github/aiiu-lab/MeDM/blob/main/colabs/gt_flow.ipynb)
- [Predicted flow](https://colab.research.google.com/github/aiiu-lab/MeDM/blob/main/colabs/pred_flow.ipynb)

## Environment setup
We use [conda](https://docs.conda.io/projects/miniconda/en/latest/)
to maintain the Python environment
```
conda env create -f environment.yml
```
The implementation of MeDM is in [pipeline_medm.py](diffusers-0.20.0/src/diffusers/pipelines/medm/pipeline_medm.py). We incorporate MeDM into the snapshot of [Diffusers](https://huggingface.co/docs/diffusers/index) at version `0.20.0`. To install it, simply use
```
cd diffusers-0.20.0
pip install .
```

## Minimal examples
We provide two simple examples on how to use `MeDMPipeline` in `colabs`. Remember to skip the colab-specific blocks when running locally.


### Citation
If you find our work useful, please consider cite this work as
```bibtex
@article{chu2023medm,
      title={MeDM: Mediating Image Diffusion Models for Video-to-Video Translation with Temporal Correspondence Guidance},
      author={Ernie Chu and Tzuhsuan Huang and Shuo-Yen Lin and Jun-Cheng Chen},
      journal={arXiv preprint arXiv:2308.10079},
      year={2023}
}
```


