# Long Term trajectory prediction

Based on paper [Long-Term On-Board Prediction of People in Traffic Scenes under Uncertainty](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3887.pdf) and github [project](https://github.com/apratimbhattacharyya18/onboard_long_term_prediction).

This project is utilizing Bayesian neural network to improove precision and  capture uncertainty in regression task (predicting future bounding box position of pedestrians from past bounding box positions). Ghahramani, Gal [[1]](https://arxiv.org/pdf/1506.02142.pdf) showed that a neural network with arbitrary depth and non-linearities, with dropout applied before every weight layer, is mathematically equivalent to an approximation to the probabilistic deep Gaussian process also known as 'MC dropout', as extension of Ghahramani, Gal work [[1]](https://arxiv.org/pdf/1506.02142.pdf), Mobiny et al.[[5]](https://arxiv.org/pdf/1906.04569.pdf) used dropconnect [[4]](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf) instead of dropout. This method is called 'MC dropconnect'. Predicting long term positions of pedestrians requires "learning" long term dependencies, this can be achieved by recurrent neural networks especialy LSTM.  This work utilizes vanilla LSTM, LSTM with applied dropout as in Gal and Ghahramani [[3]](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf) paper and dropconnect LSTM (this project). For predicting n steps to the future encoder decoder architecture is used similarly as Bhattacharyya et al. [[7]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3887.pdf). Uncertainty estimation is categorized as aletoric (noise in data) and epistemic (model uncertainty) according to Kendall, Gal work [[2]](https://arxiv.org/pdf/1703.04977.pdf). 

## Results

![final_h (2)](https://user-images.githubusercontent.com/32457553/71257791-eaa25480-2334-11ea-9175-271e4721aeba.jpg)

First and second row: bounding box prediction (blue: ground truth, red: vanilla LSTM, yellow: MC dropout, green: MC dropconnect LSTM). Third and fourth row predictive probability heat map by MC dropconnect model(red: high probability, blue: low probability) Third row predictive probability (low uncertainty). Fourth row predictive probability (high uncertainty).

![pred_gif](https://user-images.githubusercontent.com/32457553/71253631-c2acf400-2328-11ea-8c58-23e57a226af4.gif)

Frames 1 - 8 (observations), frames 9 - 23 (predictions), (blue: ground truth, red: vanilla LSTM, yellow: MC dropout, green: MC dropconnect LSTM).

![var_gif](https://user-images.githubusercontent.com/32457553/71253732-03a50880-2329-11ea-89ac-2966d6fc421b.gif)

Frames 1 - 8 (observations), frames 9 - 23 (predictions). Predictive probability heat map (MC dropconnect model).

| Model | MSE | Aletoric uncertainty | Epistemic uncertainty |
| ------ | ------ | ------ | ------ |
| Vanilla LSTM encoder decoder| 438.9 | - | - |
| MC dropout LSTM encoder decoder | 450.3 | 18.8 | 109.5 |
| MC dropconnect LSTM encoder decoder |  424.9 | 13.7 | 34.9

All values are in pixels aletoric uncertainty is scaled check train.py

## Requirments 

  - python 3.7.4
  - Tensorlflow 2.0.0
  - h5py
  - cv2

## Data
  - Pedestrian tracks files can be downloaded  [here](https://drive.google.com/drive/folders/1hOkm0O4AMrF0bNzdbY_RgOkeopE30R6U).
  - Images for tracks from [cityscapes dataset](https://www.cityscapes-dataset.com/downloads/)

## Training
### Vanilla LSTM
```
python train.py --model_name vanilla_lstm --epochs 20 --batch_size 128 --unit_dropout 0.0 --weight_dropout 0.0 --predict_variance False --mc_samples 1 --use_mc_dropout False --loss_fn mse
```
### MC dropout LSTM
```
python train.py --model_name mc_dropout_128 --epochs 40 --batch_size 128 --unit_dropout 0.25 --weight_dropout 0.0 --mc_samples 50
```
### MC dropconnect LSTM
```
python train.py --model_name mc_dropconnect --epochs 40 --batch_size 128 --unit_dropout 0.0 --weight_dropout 0.25 --mc_samples 50
```

## Todos:
  - data preparation process refactor
  - two stream model
  
## References
  - [1] Ghahramani, Gal. [Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf), 2016.
  - [2] Kendall, Gal. [What Uncertainties Do We Need in Bayesian Deep
Learning for Computer Vision?](https://arxiv.org/pdf/1703.04977.pdf), 2017.
  - [3] Gal, Ghahramani. [A Theoretically Grounded Application of Dropout in
Recurrent Neural Networks](https://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf), 2016.
  - [4] Wan et al. [Regularization of Neural Networks using DropConnect](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf), 2013
  - [5] Mobiny et al. [DropConnect Is Effective in Modeling Uncertainty
of Bayesian Deep Networks](https://arxiv.org/pdf/1906.04569.pdf), 2019
  - [6] Srivastava et al. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), 2014.
  - [7] Bhattacharyya et al. [Long-Term On-Board Prediction of People in Traffic Scenes under Uncertainty](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3887.pdf), 2018.
