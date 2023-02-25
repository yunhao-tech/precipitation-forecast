Data camp project

[task reference](https://github.com/ramp-kits/solar_wind.git)

[ramp-kit doc](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/workflow.html)

分工：

1.数据收集和清理（制作 torch dataset/dataloader）
@Zong Shang

2.模型搭建
@Yang Zhang

3.整合进 ramp api
@Yunhao

4.文档编写、starting_kit.ipynb
@Yushan Liu

5.统筹、测试及各种不便分割的边角工作
@maosicheng98

(以上内容保留至最终定稿)

---

基本目标：依据连续的 18 帧卫星雷达图像，预测接下来的连续 18 帧雷达图像，损失函数为像素间 L2 距离。

训练数据：前 18 帧为输入 X_train，后 18 帧为 Y_train

测试数据：前 18 帧为输入 X_test，后 18 帧为 Y_test

训练数据包含 2021 年 1 月前 10 天晚上 20-23 点每隔 5 分钟的雷达图像，测试数据包含 2021 年 2 月前 10 天晚上 20-23 点每隔 5 分钟的雷达图像。
在 problem.py 中注意划分前后 18 帧，即实现`split_X_Y`

由于 api 的下载限制，我们事先准备好数据，以免重新配置 api 的麻烦。
Due to the download restriction of a single api, we prepare the data for you so you don't need to run download_data.py with the trouble of reconfiguration your own api.


# Convolutional LSTM neural networks for predicting precipitation in the short-term

_Authors: Yunhao Chen, Sicheng Mao, Zong Shang, Yushan Liu_


Convolutional neural networks are a type of deep learning algorithm that have shown promising results in detecting features in images, and LSTM works pretty well on tasks involving the time dimension such as time series prediction. These networks have the ability to learn spatial and temporal dependencies in the data by integrating convolutional layers and LSTM layers, which makes them effective in capturing complex patterns in weather data. By training on historical radar reflectivity data, Convolutional LSTM neural networks can make accurate predictions of precipitation in the short-term, providing valuable information for weather forecasting and disaster preparedness.

The radar dataset which contains composite reflectivity are collected by KNMI, in a raw format of every 5 minutes. Each training data point consists of 36 consecutive radar raw files, the first 18 frames being used as features and the last 18 as the target to predict by neural network. Basically, we are trying to predict the precipitation in the next 1.5 hours given the data of the past 1.5 hours.

Training data: first 18 frames as X_train, last 18 frames as Y_train

Test data: first 18 frames as X_test, last 18 frames as Y_test

The training data consisted of radar images at 5-minute intervals from 20-23 pm for the first 10 days of January 2021, and the test data consisted of radar images at 5-minute intervals from 20-23 pm for the first 10 days of February 2021.

Due to the download restriction of a single api, we prepare the data for you so you don't need to run `download_data.py` with the trouble of reconfiguration your own api.

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
