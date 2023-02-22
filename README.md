# Neural Jokes

Recurrent neural network for creating jokes

This multi-layered recurrent neural network is capable of character-level jokes generation.

Layer (type)                Output Shape              Param #
-----------------------------------------------------------------
embedding (Embedding)       multiple                  86016

gru (GRU)                   multiple                  1575936

dropout (Dropout)           multiple                  0

gru_1 (GRU)                 multiple                  1575936

dropout_1 (Dropout)         multiple                  0

gru_2 (GRU)                 multiple                  1575936

dropout_2 (Dropout)         multiple                  0

gru_3 (GRU)                 multiple                  1575936

dropout_3 (Dropout)         multiple                  0

dense (Dense)               multiple                  86184
-----------------------------------------------------------------
Total params: 6,475,944
Trainable params: 6,475,944
Non-trainable params: 0

This NN is trained on 6 MB of data – 23092 jokes in Russian from different websites.

The neural network I trained is saved in the file *model-0.683.h5*