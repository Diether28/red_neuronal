import pymongo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model # type: ignore

# Conectar a MongoDB
client = pymongo.MongoClient("mongodb+srv://admin:1234@foro.ofeh2.mongodb.net/?retryWrites=true&w=majority&appName=Foro")
db = client["ai_assistant"]
colección = db["preguntas_respuestas"]

# Cargar datos desde MongoDB
datos = list(colección.find())
if not datos:
    raise Exception("La colección 'mi_coleccion' está vacía. Por favor, inserta datos.")

# Extraer las preguntas y respuestas
preguntas_db = [dato['Pregunta'] for dato in datos]
respuestas_db = [dato['Respuesta'] for dato in datos]

# Configuración de parámetros
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 50

# Tokenización de las preguntas y respuestas
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(preguntas_db)

# Convertir preguntas a secuencias
preguntas_secuencias = tokenizer.texts_to_sequences(preguntas_db)
preguntas_padded = pad_sequences(preguntas_secuencias, maxlen=MAX_SEQUENCE_LENGTH)

# Convertir respuestas a etiquetas
etiquetas = respuestas_db  # Las respuestas son cadenas, las mantendremos como texto

# Crear un modelo de red neuronal
model = keras.Sequential([
    layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=64, input_length=MAX_SEQUENCE_LENGTH),
    layers.LSTM(64),  # Capa LSTM para procesar secuencias de texto
    layers.Dense(64, activation="relu"),
    layers.Dense(len(etiquetas), activation="softmax")  # Salida con el tamaño de las posibles respuestas
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convertir las respuestas a números (etiquetas)
# Necesitamos convertir las respuestas a un formato numérico para entrenar el modelo
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
etiquetas_encoded = label_encoder.fit_transform(etiquetas)

# Entrenamiento del modelo
model.fit(preguntas_padded, etiquetas_encoded, epochs=10, batch_size=32)

# Guardar el modelo entrenado
model.save('model_respuestas.h5')

# Cargar el modelo entrenado
model = load_model('model_respuestas.h5')

# FastAPI configuración
app = FastAPI()

# Clase de solicitud para obtener una pregunta
class PreguntaRequest(BaseModel):
    texto: str

@app.post("/pregunta")
def responder_pregunta(request: PreguntaRequest):
    """
    Recibe una pregunta y devuelve la respuesta generada por el modelo entrenado.
    """
    pregunta = request.texto
    
    # Convertir la pregunta a secuencia numérica
    pregunta_secuencia = tokenizer.texts_to_sequences([pregunta])
    pregunta_padded = pad_sequences(pregunta_secuencia, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Predecir la respuesta
    respuesta_encoded = model.predict(pregunta_padded)
    respuesta_index = np.argmax(respuesta_encoded)
    
    # Convertir el índice a la respuesta original
    respuesta = label_encoder.inverse_transform([respuesta_index])[0]
    
    return {"Respuesta": respuesta}

@app.get("/")
def inicio():
    return {"mensaje": "API de preguntas y respuestas utilizando TensorFlow funcionando correctamente."}
