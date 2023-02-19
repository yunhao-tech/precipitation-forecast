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
