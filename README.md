# SOS: Score-based Oversampling for Tabular Data


This repo contains the official implementation for the paper [SOS: Score-based Oversampling for Tabular Data](https://arxiv.org/abs/2206.08555)

by [Jayoung Kim](jayoung.kim@yonsei.ac.kr), [Chaejeong Lee](chaejeong_lee@yonsei.ac.kr), [Yehjin Shin](yehjin.shin@gmail.com), [Sewon Park](sw0413.park@samsung.com), [Minjung Kim](mj100.kim@samsung.com), [Noseong Park](noseong@yonsei.ac.kr) and [Jihoon Cho](jihoon1.cho@samsung.com)

--------------------

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --mode: <train|fine_tune>: Running mode: train or fine_tune
  --workdir: Working directory
```

* `config` is the path to the config file. Our prescribed config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the path of a config file is a combination of the following dimensions:
  * dataset: One of `Default`, `Shoppers`, `WeatherAUS`, `Satimage`.
  * continuous: train the model with continuously sampled time steps. 

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `mode` is either "train" or "fine_tune". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints exist in `workdir/checkpoints-meta`. When set to "fine_tune", it can do fine-tune the model.

--------------------

## Pretrained checkpoints

Checkpoint for `WeatherAUS` is provided in this [Google drive](https://drive.google.com/drive/u/1/folders/0AHi5jmfSpr0VUk9PVA).


--------------------

## References

This work is built upon some previous papers which might also interest you:
* Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "Score-Based Generative Modeling through Stochastic Differential Equations." *Proceedings of the 10th Annual Conference on International Conference on Learning Representations*. 2021.
* Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Song, Yang, and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.

## License

Copyright (C) 2023 Samsung SDS Co., Ltd. All rights reserved.
Released under the Samsung SDS Public License V1.0. 
For details on the scope of licenses, please refer to the License.md file (https://github.com/JayoungKim408/SOS/blob/master/License.md).

This project was basically developed based on previous open-source codes: https://github.com/yang-song/score_sde_pytorch.

