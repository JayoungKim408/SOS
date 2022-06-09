# SOS: Score-based Oversampling for Tabular Data


This repo contains the official implementation for the paper [{SOS: Score-based Oversampling for Tabular Data}](https://openreview.net/forum?id=PxTIG12RRHS)

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
  * dataset: One of `Default`, `Shoppers`, `Surgical`, `WeatherAUS`, `Buddy`, `Satimage`.
  * continuous: train the model with continuously sampled time steps. 

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `mode` is either "train" or "fine_tune". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints exist in `workdir/checkpoints-meta`. When set to "fine_tune", it can do fine-tune the model.


## Pretrained checkpoints
Checkpoint for `WeatherAUS` is provided in this [Google drive](https://drive.google.com/drive/u/1/folders/0AHi5jmfSpr0VUk9PVA).


<!-- 
| Checkpoint path | FID | IS | FID (ODE) | NNL (bits/dim) |
|:----------|:-------:|:----------:|:----------:|:----------:|
| [`ve/cifar10_ncsnpp/`](https://drive.google.com/file/d/1fXgBupLzThTGLLsiYCHRQJixuDsR1bSI/view?usp=sharing) |  2.45 | 9.73 | - | - |
| [`ve/cifar10_ncsnpp_continuous/`](https://drive.google.com/drive/folders/1Ko7hOCMIM6yFShJCIU4LBsF0sfjWuafa?usp=sharing) | 2.38 | 9.83 | - | - |
| [`ve/cifar10_ncsnpp_deep_continuous/`](https://drive.google.com/drive/folders/1rvziylUQiXWyOF1jVhGxzgGmtp1oTNT5?usp=sharing) | **2.20** | **9.89** | - | - |
| [`vp/cifar10_ddpm/`](https://drive.google.com/drive/folders/1vzeGmgCj95_04MTDh6aa5BI098Q8ybO5?usp=sharing) | 3.24 | - | 3.37 | 3.28 |
| [`vp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1qTZXghJxo8t5gTN52FuAWO6YABu2ElTN?usp=sharing) | - | - | 3.69| 3.21 |
| [`vp/cifar10_ddpmpp`](https://drive.google.com/drive/folders/14AhlnhRryO7XqjrHEtHZRcb_v_bEUd7X?usp=sharing) | 2.78 | 9.64 | - | - |
| [`vp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1ikbUY_K4Rc2-lPz7baPxdEXtx76Xn5Ov?usp=sharing) | 2.55 | 9.58 | 3.93 | 3.16 |
| [`vp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/1F74y6G_AGqPw8DG5uhdO_Kf9DCX1jKfL?usp=sharing) | 2.41 | 9.68 | 3.08 | 3.13 |
| [`subvp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1Qk6SaMq3EFnMH1rr2OdFZU5IzDKYbskh?usp=sharing) | - | - | 3.56 | 3.05 |
| [`subvp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1tDz-jQ-tD5A_UjB1gxzoofo07W0LC1aO?usp=sharing) | 2.61 | 9.56 | 3.16 | 3.02 |
| [`subvp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/1qjKjuZULYu2uN0sP79yTPkOhEvUPhYnU?usp=sharing) | 2.41 | 9.57 | **2.92** | **2.99** |

| Checkpoint path | Samples |
|:-----|:------:|
| [`ve/bedroom_ncsnpp_continuous`](https://drive.google.com/drive/folders/1tcvR65amqrD65Hn0EPlZqNPiCZh2rg4M?usp=sharing) | ![bedroom_samples](assets/bedroom.jpeg) |
| [`ve/church_ncsnpp_continuous`](https://drive.google.com/drive/folders/1h8JayORNKTo8pCCLMr0ZDJkM7U87dKM5?usp=sharing) | ![church_samples](assets/church.jpeg) |
| [`ve/ffhq_1024_ncsnpp_continuous`](https://drive.google.com/drive/folders/1GwcthBS4Ry54eA_fIg1hOCfThQ6I3u1L?usp=sharing) |![ffhq_1024](assets/ffhq_1024.jpeg)|
| [`ve/ffhq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/1-2tUJ3tOU2AruyMYPB33x1aWVOQMSygM?usp=sharing) |![ffhq_256_samples](assets/ffhq_256.jpg)|
| [`ve/celebahq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/1LK2bGXpZBzJKLCcL_NfUKsr8dxT8XVh5?usp=sharing) |![celebahq_256_samples](assets/celebahq_256.jpg)|


 -->

<!-- 
## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
``` -->

This work is built upon some previous papers which might also interest you:
* Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "Score-Based Generative Modeling through Stochastic Differential Equations." *Proceedings of the 10th Annual Conference on International Conference on Learning Representations*. 2021.
* Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Song, Yang, and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
