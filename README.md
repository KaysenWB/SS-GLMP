# SS-GLMP

This is the implementation code for the paper "Enhancing Risk Perception by Integrating Ship Interactions in Multi-Ship Encounters: A Graph-Based Learning Method".

Implementation and performance comparison of several common deep learning networks for ship motion prediction, including LSTM, GRU, Seq2Seq, TCN, Transformer, and our network GLMP. The input data are processed AIS data containing trajectory information and adjacency matrix, processed with reference to https://github.com/KaysenWB/AIS-Process.git.

The network structure is shown below, with VGAE for spatial interactive learning and attention for time series modelling.

![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure01.jpg?raw=true)
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure02.jpeg?raw=true)

![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure03.jpg?raw=true)

![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure04.jpg?raw=true)
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure06.jpg?raw=true)

![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure08.jpeg?raw=true)
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure10.jpeg?raw=true)
