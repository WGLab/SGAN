# SGAN
This is the code of the article ***Model performance and interpretability of semi-supervised generative adversarial networks to predict oncogenic variants with unlabeled data***. In this pilot study, we present an ensemble method that tries to solve the oncogenicity prediction problem with insufficient labeled variants using semi-supervised learning model. The model takes 23 prediction scores from avaliable methods and 12 clinical evidence-based scores as input, which can be obtained by [ANNOVAR](https://annovar.openbioinformatics.org/en/latest/) and [CancerVar](https://cancervar.wglab.org/). Compared with other existing softwares and supervised learning method, the result indicates that the predictive performance is slightly improved using 4,000 labeled data and 60,000 unlabeled data, 

Here, we provided the scripts which are based on Python and Jupyter Notebook. User can use `sgan.py` to generate the model input features and predict the oncogenicity of variants data. If you are interested in the methodology development, you can have a look at `train.ipynb`, which illustrates how models are trained.

## Dependencies Installation

1. Download the repository 
```bash
git clone https://github.com/WGLab/SGAN.git
````

2. Create conda environment
```bash
conda env create -f sgan.yml
conda activate sgan
```

3. Test the enviroment

```bash
python sgan.py -h
```

## Basic Usage

We provide an example about how to use the `sgan.py` to get the feature data, and predict the oncogenicity scores of variants. 

### Example
- generate feature data;
```bash
python sgan.py convert --annovar_path example/FDA.hg19_multianno.txt.grl_p \
--cancervar_path example/FDA.hg19_multianno.txt.cancervar \
--method ensemble \
--missing_count 13 \
--database saves/nonmissing_db.npy \
--output example/FDA.hg19_multianno.features
```
- predict oncogenicity scores
```bash
python sgan.py predict --input example/FDA.hg19_multianno.features \
--cancervar_path example/FDA.hg19_multianno.txt.cancervar \
--method ensemble \
--device cpu \
--config saves/ensemble.pt \
--output example/FDA.hg19_multianno.predict
```

### Detailed Uasge

```bash
python sgan.py convert -h

usage: sgan.py convert [-h] -a ANNOVAR_PATH -c CANCERVAR_PATH [-m METHOD] [-n MISSING_COUNT] -d DATABASE -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -a ANNOVAR_PATH, --annovar_path ANNOVAR_PATH
                        the path to annovar file
  -c CANCERVAR_PATH, --cancervar_path CANCERVAR_PATH
                        the path to cancervar file
  -m METHOD, --method METHOD
                        output evs features or ensemble features (option: evs, ensemble)
  -n MISSING_COUNT, --missing_count MISSING_COUNT
                        variant with more than N missing features will be discarded, (default: 5)
  -d DATABASE, --database DATABASE
                        database for feature normalization
  -o OUTPUT, --output OUTPUT
                        the path to output
```

```bash
python sgan.py predict -h

usage: sgan.py predict [-h] -i INPUT -v CANCERVAR_PATH [-m METHOD] [-d DEVICE] -c CONFIG -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        the path to input feature
  -v CANCERVAR_PATH, --cancervar_path CANCERVAR_PATH
                        the path to cancervar file
  -m METHOD, --method METHOD
                        use evs features or ensemble features (option: evs, ensemble)
  -d DEVICE, --device DEVICE
                        device used for dl-based predicting (option: cpu, cuda)
  -c CONFIG, --config CONFIG
                        the path to trained model file
  -o OUTPUT, --output OUTPUT
                        the path to output
```


### Advanced Usage
For users interested in method development, we prvide a jupyter notebook `train.ipynb`, which is user-friendly and describes how to train a model. What's more, we can use `tensorboard` to observe the change of losses and accuracies during the training process, by running `tensorboard --logdir=runs`. The folder `runs` will be created, if the model starts traing. 



## What and Why?

Our semi-supervised model is based on unsupervised **G**enerative **A**dversarial **N**etworks (GANs). Goodfellow et. al firstly proposed [GAN](https://arxiv.org/abs/1406.2661). GANs contain 2 parts: (1) generator and (2) discriminator. In a standard training process, the generator generated fake samples from noise vectors, and then, the discriminator is to distinguish whether the input sample is fake or real. The generator is supposed to fool the discriminator, which means the generator is trained to fit the underlying probability density function of real data. 

<img src="https://github.com/WGLab/SGAN/blob/main/figs/semi-supervised.png" width="300" alt="distribution"/><br/>

In our study, we can assume that the input data (interpretation scores from different guidelines and softwares) follows an unknown distribution (grep points). Then, the discriminator is to classify the data into 3 parts: benign (red point), oncogenic (blue point), and fake. Once the generator has been trained to generate synthtic samples following the underlying distribution, the discriminator can tell us the border of benign/oncogenic class with a small number of labeled data. The detail of our model is shown following:
<img src="https://github.com/WGLab/SGAN/blob/main/figs/ourmodel.png" width="600" alt="distribution"/><br/>

## Run the code

We used conda to build environment and the code is implemented in Pytorch. You can train a model on jupyter notebook.



## Reference:

[1] https://github.com/etaoxing/semi-supervised-gan

[2] https://github.com/opetrova/SemiSupervisedPytorchGAN

[3] https://arxiv.org/abs/1606.03498
