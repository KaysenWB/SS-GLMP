# SS-GLMP

This is the implementation code for the paper "Enhancing Risk Perception by Integrating Ship Interactions in Multi-Ship Encounters: A Graph-Based Learning Method".

Implementation and performance comparison of several common deep learning networks for ship motion prediction, including LSTM, GRU, Seq2Seq, TCN, Transformer, and our network GLMP. The input data are processed AIS data containing trajectory information and adjacency matrix, processed with reference to https://github.com/KaysenWB/AIS-Process.git.

![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure01.jpg?raw=true)

The network structure is shown below, with VGAE for spatial interactive learning and attention for time series modelling.
![Figure](https://github.com/KaysenWB/GCN-Transformer/blob/main/Figure.jpeg?raw=true)
