The goal of this project is to train a SpecAugment + LibriSpeech 960h + Listen, Attend, and Spell pipeline as in the SpecAugment paper, and then run it in real time on this GPU here.

Encoder:

* two layers of 3x3 convolution with stride 2
* three or four layers of bidirectional LSTM, with a projection layer and batch normalization layer each (a projection layer is presumably projecting the channels down??? or "subsampling" from adjacent times???)

Decoder:

* one or two LSTM
* it uses attention

$$c_k = \sum_l \alpha_{kl} h_l, \quad \alpha_{kl} = softmax(f_1(h_l)^T f_2(o_k^1))$$

SpecAugment: https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html

Listen, Attend, and Spell: https://arxiv.org/pdf/1508.01211.pdf and https://arxiv.org/pdf/1902.01955.pdf

## Other papers:

Very Deep Convolutional Networks for End-to-End Speech Recognition: https://arxiv.org/pdf/1610.03022.pdf

Sequence-to-Sequence Models Can Directly Translate Foreign Speech: https://arxiv.org/pdf/1703.08581.pdf
