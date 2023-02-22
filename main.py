import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from model import *
from config import *

with open(FILE, 'r', encoding='utf-8') as f:
    text = f.read()
    data = text.split('¶')

filtered_data = [START_SIGN + obj[1:] for obj in data if len(obj) <= SEQ_LEN]
print(len(filtered_data))

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
    filters='',
    lower=False,
    split=''
)

tokenizer.fit_on_texts([STOP_SIGN])
tokenizer.fit_on_texts(filtered_data)

vocabulary_size = len(tokenizer.word_counts) + 1
print(vocabulary_size)

def sequence_to_string(sequence):
    stringified = tokenizer.sequences_to_texts([sequence])[0]
    return stringified

vectorized_data = tokenizer.texts_to_sequences(filtered_data)

padded_vectorized_data_without_stops = tf.keras.preprocessing.sequence.pad_sequences(
    vectorized_data,
    padding='post',
    truncating='post',
    maxlen=SEQ_LEN - 1,
    value=tokenizer.texts_to_sequences([STOP_SIGN])[0]
)

padded_vectorized_data = tf.keras.preprocessing.sequence.pad_sequences(
    padded_vectorized_data_without_stops,
    padding='post',
    truncating='post',
    maxlen=SEQ_LEN + 1,
    value=tokenizer.texts_to_sequences([STOP_SIGN])[0]
)

dataset = tf.data.Dataset.from_tensor_slices(padded_vectorized_data)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = dataset.map(split_input_target)

dataset = (
    dataset
    .shuffle(10000)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

model = Prototype(
    vocab_size=vocabulary_size,
    embedding_dim=EMBEDDED_DIM,
    rnn_units=RNN_UNITS,
    do=DO)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss = SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])

# TRAIN MODEL

class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights("%d-%s.h5" % (epoch, str(round(logs['loss'], 3))))

checkpoint = ModelCheckpoint()
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint])

# RUN MODEL

model.load_weights('model-0.683.h5')
model = Model(model)

next_char = tf.constant(np.array(tokenizer.texts_to_sequences([START_SIGN])))
states = [None] * 4
result = str()

for i in range(SEQ_LEN):
    next_char, states = model.predict(next_char, states=states, random=2)
    next_symbol = tokenizer.sequences_to_texts([[next_char]])[0]
    if next_symbol == STOP_SIGN:
        break
    print(next_symbol, end='')
    next_char = tf.constant([[next_char]])