## Image-to-LaTeX Converter

This project implements a model that converts images of mathematical expressions into LaTeX code using an attention-based encoder-decoder architecture in PyTorch.

The dataset used for training and evaluation is available at:

[im2markup.yuntiandeng.com/data/](http://im2markup.yuntiandeng.com/data/)


This project is inspired by the following papers:

-  **[Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks (Deng et al., 2016)](http://arxiv.org/pdf/1609.04938v1.pdf)**  
- **[Image-to-Markup Generation with Coarse-to-Fine Attention (Xu et al., 2019)](https://arxiv.org/abs/1908.11415)**
- **[Image to Latex](https://cs231n.stanford.edu/reports/2017/pdfs/815.pdf)**


To run the model, install the dataset from the link above and decompress the images folder. Then add the paths required in the respective training, validation and utils scripts. The vocabulary and json files have already been provided.