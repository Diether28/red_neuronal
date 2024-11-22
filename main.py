import pymongo
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Conectar a MongoDB
client = pymongo.MongoClient("mongodb+srv://admin:1234@foro.ofeh2.mongodb.net/?retryWrites=true&w=majority&appName=Foro")
db = client["ai_assistant"]
colección = db["preguntas_respuestas"]

# Cargar datos desde MongoDB
datos = list(colección.find())
if not datos:
    raise Exception("La colección 'mi_coleccion' está vacía. Por favor, inserta datos.")

# Preprocesamiento de las preguntas y respuestas
# preguntas_db = [dato['Pregunta'] for dato in datos]
# respuestas_db = [dato['Respuesta'] for dato in datos]

# Verificar que los documentos contienen los campos correctos
preguntas_db = []
respuestas_db = []
for dato in datos:
    if 'Pregunta' not in dato or 'Respuesta' not in dato:
        raise Exception(f"Documento incorrecto encontrado: {dato}")
    preguntas_db.append(dato['Pregunta'])  # Asegúrate de que 'Pregunta' es el campo correcto
    respuestas_db.append(dato['Respuesta'])  # Asegúrate de que 'Respuesta' es el campo correcto

# Inicializar el vectorizador TF-IDF
vectorizer = TfidfVectorizer()

# Ajustar el vectorizador con las preguntas existentes en la base de datos
vectorizer.fit(preguntas_db)

# Crear la aplicación FastAPI
app = FastAPI()

# Definir el modelo de solicitud para la pregunta
class PreguntaRequest(BaseModel):
    texto: str

@app.post("/pregunta")
def responder_pregunta(request: PreguntaRequest):
    """
    Recibe una pregunta y devuelve la respuesta más similar de la base de datos.
    """
    pregunta = request.texto
    
    # Convertir la pregunta recibida en un vector TF-IDF
    pregunta_vector = vectorizer.transform([pregunta])
    
    # Calcular la similitud del coseno entre la pregunta recibida y las preguntas en la base de datos
    similitudes = cosine_similarity(pregunta_vector, vectorizer.transform(preguntas_db))
    
    # Obtener el índice de la pregunta más similar
    indice_similar = np.argmax(similitudes)
    
    # Devolver la respuesta correspondiente
    respuesta = respuestas_db[indice_similar]
    
    return {"Respuesta": respuesta}

@app.post("/agregar")
def agregar_dato(request: PreguntaRequest):
    """
    Agrega una nueva pregunta y su respuesta correspondiente a la base de datos.
    """
    pregunta = request.texto
    # Respuesta simple de ejemplo. Deberías usar un modelo real para generar respuestas aquí.
    respuesta = "Respuesta no disponible para esta pregunta."
    
    # Insertar pregunta y respuesta en MongoDB
    colección.insert_one({"Pregunta": pregunta, "Respuesta": respuesta, "Categoría": "Desconocido"})
    
    return {"mensaje": "Pregunta y respuesta agregadas exitosamente", "respuesta": respuesta}

@app.get("/preguntas")
def obtener_preguntas():
    """
    Devuelve todas las preguntas y respuestas almacenadas en la base de datos.
    """
    datos = list(colección.find({}, {"_id": 0}))  # No mostrar el _id
    return {"preguntas": datos}

@app.get("/")
def inicio():
    """
    Ruta de inicio para comprobar que la API está funcionando.
    """
    return {"mensaje": "API de preguntas y respuestas funcionando correctamente."}
