# Convolutional LSTM neural networks for predicting precipitation in the short-term

_Authors: Yunhao Chen, Sicheng Mao, Zong Shang, Yushan Liu_


Convolutional neural networks are a type of deep learning algorithm that have shown promising results in detecting features in images, and LSTM works pretty well on tasks involving the time dimension such as time series prediction. These networks have the ability to learn spatial and temporal dependencies in the data by integrating convolutional layers and LSTM layers, which makes them effective in capturing complex patterns in weather data. By training on historical radar reflectivity data, Convolutional LSTM neural networks can make accurate predictions of precipitation in the short-term, providing valuable information for weather forecasting and disaster preparedness.

The radar dataset which contains composite reflectivity are collected by KNMI, in a raw format of every 5 minutes. Each training data point consists of 36 consecutive radar raw files, the first 18 frames being used as features and the last 18 as the target to predict by neural network. Basically, we are trying to predict the precipitation in the next 1.5 hours given the data of the past 1.5 hours.


#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install ramp-workflow
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](precipitation_forecast_starting_kit.ipynb).

To test the starting-kit, run


```
ramp-test --quick-test
```


#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](https://ramp.studio) ecosystem.
