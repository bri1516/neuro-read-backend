import os
import json
import re # <--- AGREGADO: Herramienta para buscar el JSON aunque la IA mande texto extra
import logging # <--- AGREGADO: Para ver qué hace la app desde la consola de Render
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# <--- AGREGADO: Configuración para que el servidor imprima avisos útiles
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroRead API", description="Motor NLP para terapia de lectura y fluidez")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.error("¡ATENCIÓN! No se encontró la API KEY de Gemini.") # <--- AGREGADO: Aviso si falta la llave

# --- MODELOS DE DATOS ---
class PeticionEjercicio(BaseModel):
    perfil: str
    nivel: int  # 0 será el nivel de diagnóstico inicial

class PeticionAnalisis(BaseModel):
    perfil: str
    nivel: int
    texto_original: str
    texto_leido: str # La transcripción que capturó tu motor de voz en Flutter

class PeticionRespiracion(BaseModel):
    tipo: str

# <--- AGREGADO: Función salvavidas por si Gemini no devuelve un JSON perfecto
def extraer_json_seguro(texto_sucio: str):
    try:
        # Busca todo lo que esté entre llaves { } ignorando el resto
        match = re.search(r'\{.*\}', texto_sucio, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.error(f"Error al forzar lectura de JSON: {e}")
    return None

# ---------------------------------------------------------
# FASE 1: GENERACIÓN DEL EJERCICIO Y DIAGNÓSTICO
# ---------------------------------------------------------
@app.post("/generar_ejercicio")
async def generar_ejercicio(datos: PeticionEjercicio):
    logger.info(f"Petición recibida - Generar Ejercicio: Perfil {datos.perfil}, Nivel {datos.nivel}") # <--- AGREGADO: Registro

    model = genai.GenerativeModel(
    'gemini-1.5-flash', 
    generation_config={"temperature": 0.85} # <--- DÉJALO, esto da variedad a los cuentos
)
  
    
    # Lógica de Especialización por Perfil
    if datos.perfil == "infantil":
        contexto_rol = """
        Eres un cuentacuentos y terapeuta infantil de lenguaje. Tu tono es mágico, divertido y cálido. 
        Usa temáticas de animales, dinosaurios, magia o juegos. Lenguaje muy sencillo.
        """
        niveles_dificultad = {
            0: "EVALUACIÓN INICIAL: Un texto muy corto (15 palabras) que incluya una variedad de sonidos (vocales, sibilantes, nasales) para detectar dónde están las trabas del niño.",
            1: "Vocales Mágicas: Palabras aisladas que empiecen con vocales o sonidos nasales muy suaves (m, n). Máximo 5 palabras.",
            2: "Sílabas Saltofinas: Frases cortas de 3 palabras usando sílabas directas simples (la, ma, pa).",
            3: "Primeras Palabras: Oraciones de 4 a 5 palabras de uso común. Sin consonantes trabadas.",
            4: "Frases de Cristal: Frases cortas que incluyen comas obligatorias para forzar una pausa de respiración.",
            5: "El Gran Lector: Lectura rítmica. Un texto de 2 líneas con una métrica constante para mantener el ritmo.",
            6: "Cuentos Cortos: Un párrafo de 3 líneas sobre una pequeña aventura. Fomenta unir palabras fluidamente.",
            7: "Rimas Divertidas: Poemas cortos o trabalenguas infantiles muy suaves.",
            8: "Habla Veloz: Diálogo corto entre dos personajes.",
            9: "Pequeño Orador: Explicar cómo funciona algo sencillo. 4 líneas.",
            10: "Maestro de Magia: Un cuento de 5 líneas con diferentes emociones para practicar entonación."
        }
    else:
        contexto_rol = """
        Eres un coach profesional de oratoria y comunicación para adultos. Tu tono es respetuoso, motivador y estructurado.
        Usa temáticas del entorno laboral, tecnología, cultura o situaciones sociales reales.
        """
        niveles_dificultad = {
            0: "EVALUACIÓN INICIAL: Un párrafo de presentación estándar (20 palabras) para medir la fluidez base, ritmo y control de pausas del usuario.",
            1: "Inicio Suave: Palabras aisladas o frases de 2 palabras con sonidos iniciales relajados.",
            2: "Sílabas de Seda: Oraciones cortas (4 palabras) enfocadas en la transición suave entre consonante y vocal.",
            3: "Frases Cotidianas: Párrafo de 2 líneas simulando pedir un café o saludar a un vecino.",
            4: "Control de Pausas: Textos con puntuación abundante para respetar las pausas para inhalar.",
            5: "Lectura Fluida: Un texto informativo de 3 líneas sobre un tema de cultura general.",
            6: "Conversación Social: Un diálogo simulado (ej. presentarse en una fiesta).",
            7: "Entorno Laboral: Simular leer un reporte corto o dar una actualización de proyecto.",
            8: "Manejo de Estrés: Texto que simula una situación de ligera presión.",
            9: "Discurso Público: Un fragmento de discurso persuasivo corto.",
            10: "Maestro de Fluidez: Simulación de una entrevista de trabajo compleja."
        }

    objetivo_tecnico = niveles_dificultad.get(datos.nivel, "Lectura libre para practicar fluidez.")

    # El Prompt de Generación
    prompt = f"""
    {contexto_rol}
    
    El usuario está en el NIVEL {datos.nivel}. 
    (Nota: Si el nivel es 0, es su primera vez en la app y este es un texto de diagnóstico para calibrar su nivel).
    
    Genera un ejercicio único adaptado a este nivel.
    Objetivo técnico estricto: {objetivo_tecnico}
    
    Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta:
    {{
        "texto": "El contenido que el usuario debe leer.",
        "guia": "Instrucción previa a la lectura (ej. 'Inhala profundo antes de empezar')."
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        res_limpia = response.text.replace('```json', '').replace('```', '').strip()
        
        # <--- AGREGADO: Intenta leerlo normal, si falla, usa el salvavidas
        try:
            resultado = json.loads(res_limpia)
        except json.JSONDecodeError:
            resultado = extraer_json_seguro(res_limpia)
            if not resultado:
                raise ValueError("No se pudo extraer JSON de la respuesta.")
                
        logger.info("Ejercicio generado con éxito.") # <--- AGREGADO
        return resultado
        
    except Exception as e:
        logger.error(f"Error generando ejercicio: {str(e)}") # <--- AGREGADO
        return {
            "texto": "Inhala, exhala y lee a tu propio ritmo.", 
            "guia": "Tómate tu tiempo, no hay prisa."
        }

# ---------------------------------------------------------
# FASE 2: ANÁLISIS FINAL Y RECOMENDACIONES (NLP)
# ---------------------------------------------------------
@app.post("/analizar_ejercicio")
async def analizar_ejercicio(datos: PeticionAnalisis):
    logger.info(f"Petición recibida - Analizar Ejercicio. Original: '{datos.texto_original[:20]}...' Leído: '{datos.texto_leido[:20]}...'") # <--- AGREGADO

    model = genai.GenerativeModel(
    'gemini-1.5-flash', 
    generation_config={"temperature": 0.3} # <--- DÉJALO, esto lo hace más preciso al evaluar
)
    
    # Ajustamos el tono de la retroalimentación
    if datos.perfil == "infantil":
        tono = "Usa un tono muy alentador, celebrando el esfuerzo del niño como si hubieras ganado un juego. Usa lenguaje simple."
    else:
        tono = "Usa un tono profesional, objetivo y constructivo. Sé directo sobre las áreas de mejora."

    # Si es el Nivel 0 (Diagnóstico), el enfoque del análisis cambia
    contexto_evaluacion = ""
    if datos.nivel == 0:
        contexto_evaluacion = """
        ESTE ES EL DIAGNÓSTICO INICIAL DEL USUARIO.
        Calcula qué tan bien lo hizo y, en la recomendación, sugiérele en qué nivel del 1 al 10 debería empezar su entrenamiento.
        """

    prompt_analisis = f"""
    Actúa como un terapeuta de lenguaje experto en tartamudez y fluidez.
    {tono}
    
    Texto que el usuario debía leer: "{datos.texto_original}"
    Texto que el reconocimiento de voz capturó: "{datos.texto_leido}"
    
    {contexto_evaluacion}
    
    Analiza la diferencia entre ambos textos. Considera que las omisiones, repeticiones de sílabas o silencios detectados por el micrófono son indicadores de disfluencia.
    
    Devuelve ÚNICAMENTE un objeto JSON con esta estructura exacta:
    {{
        "precision": Un número entero del 0 al 100 estimando su precisión y fluidez,
        "analisis": "Un párrafo corto explicando qué hizo bien y dónde hubo trabas",
        "consejo": "Una recomendación técnica específica para su próxima lectura",
        "nivel_recomendado": Solo si el nivel actual era 0, coloca aquí el número de nivel (1-10) sugerido. Si no, devuelve null.
    }}
    """
    
    try:
        response = model.generate_content(prompt_analisis)
        res_limpia = response.text.replace('```json', '').replace('```', '').strip()
        
        # <--- AGREGADO: Mismo salvavidas para garantizar que Flutter reciba el JSON correcto
        try:
            resultado = json.loads(res_limpia)
        except json.JSONDecodeError:
            resultado = extraer_json_seguro(res_limpia)
            if not resultado:
                raise ValueError("No se pudo procesar el análisis.")
        
        # <--- AGREGADO: Asegurar que precision siempre sea un número para que Flutter no crashee
        if isinstance(resultado.get("precision"), str):
            resultado["precision"] = int(''.join(filter(str.isdigit, resultado["precision"])) or 0)
            
        logger.info(f"Análisis exitoso. Precisión: {resultado.get('precision')}%") # <--- AGREGADO
        return resultado
        
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}") # <--- AGREGADO
        return {
            "precision": 0,
            "analisis": "Hubo un problema al procesar el audio, pero lo importante es seguir practicando.",
            "consejo": "Asegúrate de estar en un lugar silencioso la próxima vez.",
            "nivel_recomendado": None
        }

# ---------------------------------------------------------
# FASE 3: MÓDULO DE RESPIRACIÓN Y BIENESTAR
# ---------------------------------------------------------
@app.post("/obtener_respiracion")
async def obtener_respiracion(datos: PeticionRespiracion):
    tecnicas = {
        "ansiedad": {
            "nombre": "Técnica 4-7-8", 
            "pasos": "Inhala por 4 segundos, mantén el aire 7 segundos, y exhala lentamente por 8 segundos.", 
            "objetivo": "Esta técnica actúa como un tranquilizante natural para el sistema nervioso."
        },
        "estres": {
            "nombre": "Respiración Cuadrada", 
            "pasos": "Inhala 4 seg, mantén 4 seg, exhala 4 seg, mantén vacío 4 seg.", 
            "objetivo": "Ideal para recuperar la concentración y reducir la tensión física."
        },
        "enfoque": {
            "nombre": "Respiración Rítmica", 
            "pasos": "Inhala 3 seg, mantén 1 seg, exhala 3 seg, mantén 1 seg.", 
            "objetivo": "Genera claridad mental y prepara las cuerdas vocales."
        }
    }
    return tecnicas.get(datos.tipo.lower(), tecnicas["ansiedad"])
