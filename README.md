
<h1 align="center"> UCGM: Unified Continuous Generative Models </h1>

<p align="center">
  Peng&nbsp;Sun<sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  Yi&nbsp;Jiang<sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://tlin-taolin.github.io/" target="_blank">Tao&nbsp;Lin</a><sup>1</sup> &ensp; &ensp;
</p>

<p align="center">
  <sup>1</sup>Westlake University &emsp; <sup>2</sup>Zhejiang University&emsp; <br>
</p>

<p align="center">
<a href="https://huggingface.co/sp12138sp/UCGM">:robot: Models</a> &ensp;
<a href="https://arxiv.org/abs/2505.07447">:page_facing_up: Paper</a> &ensp;
<a href="#label-bibliography">:label: BibTeX</a> &ensp;
  <br><br>
<a href="https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=unified-continuous-generative-models"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unified-continuous-generative-models/image-generation-on-imagenet-256x256" alt="PWC"></a> <a href="https://paperswithcode.com/sota/image-generation-on-imagenet-512x512?p=unified-continuous-generative-models"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unified-continuous-generative-models/image-generation-on-imagenet-512x512" alt="PWC"></a>
</p>

Official PyTorch implementation of **UCGM**: A unified framework for training, sampling, and understanding continuous generative models (diffusion, flow-matching, consistency models).

## :trophy: Key Results

<div align="center">
  <img src="assets/fig1_a_512.png" width="48%">
  <img src="assets/fig1_b_512.png" width="48%">
  <p>
    <strong>Generated samples from two 675M diffusion transformers trained with UCGM on ImageNet-1K 512×512.</strong><br>
    Left: A multi-step model (Steps=40, FID=1.48) | Right: A few-step model (Steps=2, FID=1.75)<br>
    <em>Samples generated without classifier-free guidance or other guidance techniques.</em>
  </p>
</div>

## :sparkles: Features

- :rocket: **Unified Framework**: Train/sample diffusion, flow-matching, and consistency models in one system  
- :electric_plug: **Plug-and-Play Acceleration**: UCGM-S boosts *pre-trained models*—e.g., given a model from [REPA-E](https://github.com/End2End-Diffusion/REPA-E) (on ImageNet 256×256), **cuts 84% sampling steps (NFE=500 → NFE=80) while improving FID (1.26 → 1.06)**  
- :1st_place_medal: **SOTA Performance**: UCGM-T-trained models outperform peers at low steps (**1.21 FID @ 30 steps on ImageNet 256×256, 1.48 FID @ 40 steps on 512×512**)  
- :zap: **Few-Step Mastery**: Just **2 steps**? Still strong (**1.42 FID on 256×256, 1.75 FID on 512×512**)  
- :no_entry_sign: **Guidance-Free**: No classifier-free guidance for UCGM-T-trained models, **simpler and faster**  
- :building_construction: **Architecture & Dataset Flexibility**: Compatible with diverse datasets (ImageNet, CIFAR, etc.) and VAEs/neural architectures (CNNs, Transformers)  
- :book: Check more features in our [paper](https://arxiv.org/abs/2505.07447)!

## :wrench: Preparation

1. Download necessary files from [Huggingface](https://huggingface.co/sp12138sp/UCGM/tree/main), including:
   - Checkpoints of various VAEs
   - Statistic files for datasets
   - Reference files for FID calculation

2. Place the downloaded `outputs` and `buffers` folders at the same directory level as this `README.md`

3. For dataset preparation (skip if not training models), run:
```bash
bash scripts/data/in1k256.sh
```

## :fast_forward: UCGM-S: Plug-and-Play Acceleration

Accelerate any continuous generative model (diffusion, flow-matching, etc.) with UCGM-S. Results marked with :zap: denote UCGM-S acceleration.  
*NFE = Number of Function Evaluations (sampling computation cost)*

| Method                                                  | Model Size | Dataset  | Resolution      | NFE           | FID  | NFE (⚡)    | FID (⚡) | Model                                                   |
| ------------------------------------------------------- | ---------- | -------- | --------------- | ------------- | ---- | ------------- | ------- | ------------------------------------------------------- |
| [REPA-E](https://github.com/End2End-Diffusion/REPA-E) | 675M       | ImageNet | 256×256         | 250×2         | 1.26 | 40×2          | 1.06    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [Lightning-DiT](https://github.com/hustvl/LightningDiT) | 675M       | ImageNet | 256×256         | 250×2         | 1.35 | 50×2          | 1.21    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DDT](https://github.com/MCG-NJU/DDT)                   | 675M       | ImageNet | 256×256         | 250×2         | 1.26 | 50×2          | 1.27    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [EDM2-S](https://github.com/NVlabs/edm2)                | 280M       | ImageNet | 512×512         | 63            | 2.56 | 40            | 2.53    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [EDM2-L](https://github.com/NVlabs/edm2)                | 778M       | ImageNet | 512×512         | 63            | 2.06 | 50            | 2.04    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [EDM2-XXL](https://github.com/NVlabs/edm2)              | 1.5B       | ImageNet | 512×512         | 63            | 1.91 | 40            | 1.88    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DDT](https://github.com/MCG-NJU/DDT)                   | 675M       | ImageNet | 512×512         | 250×2         | 1.28 | 150×2         | 1.24    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |

**:computer: Usage Examples**: Generate images and evaluate FID using a REPA-E trained model:
```bash
bash scripts/run_eval.sh ./configs/sampling_multi_steps/in1k256_sit_xl_repae_linear.yaml
```

## :gear: UCGM-T: Unified Training Framework

Train multi-step and few-step models (diffusion, flow-matching, consistency) with UCGM-T. All models sample efficiently without guidance.

| Encoders                                                                            | Model Size | Resolution  | Dataset  | NFE | FID  | Model                                                   |
| ----------------------------------------------------------------------------------- | ---------- | ----------- | -------- | --- | ---- | ------------------------------------------------------- |
| [VA-VAE](https://github.com/hustvl/LightningDiT)                                    | 675M       | 256×256     | ImageNet | 30  | 1.21 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [VA-VAE](https://github.com/hustvl/LightningDiT)                                    | 675M       | 256×256     | ImageNet | 2   | 1.42 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DC-AE](https://github.com/mit-han-lab/efficientvit/tree/master/applications/dc_ae) | 675M       | 512×512     | ImageNet | 40  | 1.48 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DC-AE](https://github.com/mit-han-lab/efficientvit/tree/master/applications/dc_ae) | 675M       | 512×512     | ImageNet | 2   | 1.75 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |

**:computer: Usage Examples**

Generate Images:
```bash
# Generate samples using our pretrained few-step model
bash scripts/run_eval.sh ./configs/training_few_steps/in1k256_tit_xl_vavae.yaml
```

Train Models:
```bash
# Train a new multi-step model (full training)
bash scripts/run_train.sh ./configs/training_multi_steps/in1k512_tit_xl_dcae.yaml

# Convert to few-step model (requires pretrained multi-step checkpoint)
bash scripts/run_train.sh ./configs/training_few_steps/in1k512_tit_xl_dcae.yaml
```

:exclamation: **Note for few-step training**:
1. Requires initialization from a multi-step checkpoint
2. Prepare your checkpoint file with both `model` and `ema` keys:
   ```python
   {
     "model": multi_step_ckpt["ema"], 
     "ema": multi_step_ckpt["ema"]
   }
   ```


## :label: Bibliography

If you find this repository helpful for your project, please consider citing our work:

```
@article{sun2025unified,
  title = {Unified continuous generative models},
  author = {Sun, Peng and Jiang, Yi and Lin, Tao},
  journal = {arXiv preprint arXiv:2505.07447},
  year = {2025},
  url = {https://arxiv.org/abs/2505.07447},
  archiveprefix = {arXiv},
  eprint = {2505.07447},
  primaryclass = {cs.LG}
}
```


## :page_facing_up: License

Apache License 2.0 - See [LICENSE](LICENSE) for details.