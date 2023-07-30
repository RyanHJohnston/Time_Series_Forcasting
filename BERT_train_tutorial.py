!pip install transformers -q
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizer
import tensorflow as tf
import sklearn.model_selection as ms
import numpy as np
import matplotlib.pyplot as plt

base = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizer.from_pretrained(base)
model = TFDistilBertForSequenceClassification.from_pretrained(base)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

X_train, X_val, y_train, y_val = ms.train_test_split(dataset.Tweet.values, dataset.Label.values, test_size=0.2)

def batch_encode(X, tokenizer):
	return tokenizer.batch_encode_plus(
		X,
		max_length=64,
		add_special_tokens=True,
		return_attention_mask=True,
		return_token_type_ids=False,
		padding=True,
    truncation=True,
		return_tensors='tf'
	)

X_train = batch_encode(X_train, tokenizer)
X_val = batch_encode(X_val, tokenizer)

y_train = y_train.reshape(y_train.shape[0], -1)
y_train = np.c_[y_train, 1 - y_train]
y_train = np.asarray(y_train).astype('float32')
y_val = y_val.reshape(y_val.shape[0], -1)
y_val = np.c_[y_val, 1 - y_val]
y_val = np.asarray(y_val).astype('float32')

tf.config.run_functions_eagerly(True)
history = model.fit(
	x=X_train.values(),
	y=y_train,
	validation_data=(X_val.values(), y_val),
	epochs=3,
	batch_size=32
)

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
