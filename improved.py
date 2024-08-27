## Universida del Valle de Guatemala
## Deep Learning
## Laboratorio #5
## 26/08/2024
## Modelo LSTM mejorado

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load and preprocess the data
max_features = 2000  # Incrementamos la cantidad de palabras
maxlen = 500         # Incrementamos el largo de las reseñas

# Cargar y preprocesar los datos
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Definir el modelo LSTM
model = Sequential()
model.add(Embedding(max_features, 64))  # Incrementamos las dimensiones de embedding
model.add(LSTM(64, return_sequences=True))  # Añadimos más unidades LSTM y return_sequences
model.add(LSTM(32))  # Añadimos una segunda capa LSTM
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo y guardar la historia del entrenamiento
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Predecir y mostrar la matriz de confusión
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, (predictions > 0.5).astype(int))
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') 
plt.close()  

# Graficar y guardar la pérdida a lo largo del tiempo
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_over_time.png')
plt.close()
