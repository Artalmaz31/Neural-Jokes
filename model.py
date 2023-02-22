import tensorflow as tf
import keras
import numpy as np

class Prototype(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, do):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru_1 = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.dropout_1 = tf.keras.layers.Dropout(do)
        self.gru_2 = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.dropout_2 = tf.keras.layers.Dropout(do)
        self.gru_3 = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.dropout_3 = tf.keras.layers.Dropout(do)
        self.gru_4 = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.dropout_4 = tf.keras.layers.Dropout(do)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states = [None] * 4, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)

        if all([state is None for state in states]):
            states[0] = self.gru_1.get_initial_state(x)
            states[1] = self.gru_2.get_initial_state(x)
            states[2] = self.gru_3.get_initial_state(x)
            states[3] = self.gru_4.get_initial_state(x)

        x, states[0] = self.gru_1(x, initial_state=states[0], training=training)
        x = self.dropout_1(x)
        x, states[1] = self.gru_2(x, initial_state=states[1], training=training)
        x = self.dropout_2(x)
        x, states[2] = self.gru_3(x, initial_state=states[2], training=training)
        x = self.dropout_3(x)
        x, states[3] = self.gru_4(x, initial_state=states[3], training=training)
        x = self.dropout_4(x)

        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

tf.config.run_functions_eagerly(True)
class Model(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function
    def predict(self, input_ids, states = [None] * 4, random = 1):
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits, indices = tf.math.top_k(predicted_logits.numpy(), k=random, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(predicted_logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        predicted_id = np.random.choice(indices[0], p=preds[0])
        return predicted_id, states