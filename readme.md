# RAMP starting kit on solar wind classification

_Authors: Yunhao Chen, Sicheng Mao, Zong Shang, Yushan Liu_

Interplanetary Coronal Mass Ejections (ICMEs) occur due to magnetic instabilities in the Sun's atmosphere and can affect the planetary environment, causing strong particle acceleration, geomagnetic storms, and geomagnetic induced currents. These effects can have serious consequences for space and ground technologies, and understanding them is an important aspect of the space weather discipline.

Signatures of ICMEs can be observed through in-situ spacecraft measurements as patterns in time series of various factors such as the magnetic field, particle density, bulk velocity, and temperature. While these patterns are recognizable to experts, their characteristics can vary widely, making automated detection a challenge.

This RAMP aims at identifing Interplanetary Coronal Mass Ejections (ICMEs) in the data collected by in-situ spacecraft.
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
