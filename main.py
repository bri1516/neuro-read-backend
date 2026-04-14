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

    model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"temperature": 0.85})
  
    # Lógica de Especialización por Perfil
    if datos.perfil == "infantil":
        contexto_rol = """ 
        Eres un guía amigable, cuentacuentos y experto en el desarrollo del lenguaje infantil (para niños de 3 a 10 años). 
        Tu tono es motivador, claro, divertido y muy paciente. No hables como un médico, sino como un maestro de escuela primaria muy querido.
        Utiliza temáticas que atrapen la atención de los niños: animales curiosos, dinosaurios amigables, el espacio, superhéroes cotidianos, la naturaleza o inventos divertidos.
        Tu objetivo es generar textos que ayuden al niño a practicar su fluidez al hablar, reduciendo la ansiedad y el tartamudeo.
        """
        
        niveles_dificultad = {
            0: """EVALUACIÓN INICIAL: 
            Objetivo: Detectar trabas en diferentes sonidos. 
            Instrucción: Escribe un párrafo único de máximo 25 a 30 palabras. Debe ser una historia sencilla (ej. un perro que persigue una pelota) pero que incluya obligatoriamente una mezcla de todos los sonidos: vocales (a, e, i, o, u), sonidos suaves (m, n, s, f) y sonidos fuertes (p, t, k, d, b). Usa oraciones cortas.""",

            1: """Nivel 1 - Sonidos Suaves (Calentamiento): 
            Objetivo: Iniciar el habla sin tensión. 
            Instrucción: Genera una lista de 5 palabras sueltas o frases de solo 2 palabras. TODAS deben empezar estrictamente con vocales o con las consonantes suaves 'm' o 'n' (ejemplo: 'el oso', 'mi mono', 'un avión'). Evita por completo letras explosivas como p, t, k, c, q.""",

            2: """Nivel 2 - Pasos Pequeños (Sílabas simples): 
            Objetivo: Unir palabras con facilidad. 
            Instrucción: Escribe 4 oraciones muy cortas (máximo 4 palabras por oración). Usa estructuras sencillas de sujeto y verbo (ejemplo: 'El gato bebe leche'). Evita palabras largas de más de 3 sílabas y NO uses grupos consonánticos complejos (nada de tr, pl, bl, cr, pr).""",

            3: """Nivel 3 - Frases Cotidianas: 
            Objetivo: Fluidez en el vocabulario del día a día. 
            Instrucción: Escribe 4 oraciones de 5 a 6 palabras relacionadas con rutinas diarias (comer, jugar, ir a la escuela). Las palabras deben ser de uso muy común para un niño. Mantén una estructura directa y fácil de leer.""",

            4: """Nivel 4 - El Semáforo (Pausas de respiración): 
            Objetivo: Enseñar al niño a respirar en medio de la frase. 
            Instrucción: Escribe 3 oraciones de longitud media (7 a 9 palabras). Es OBLIGATORIO incluir una coma (,) exactamente a la mitad de cada oración para forzar una pausa de respiración. Ejemplo: 'El perro grande, corre por el parque.'""",

            5: """Nivel 5 - Lectura Rítmica: 
            Objetivo: Mantener un ritmo constante al hablar. 
            Instrucción: Escribe un poema muy corto o un texto de 4 líneas que tenga una métrica y ritmo repetitivo. Las rimas deben ser simples y predecibles (canciones de cuna o rimas infantiles básicas). Esto ayuda a que el cerebro anticipe la siguiente palabra y reduzca el tartamudeo.""",

            6: """Nivel 6 - Pequeñas Aventuras (Coarticulación): 
            Objetivo: Leer un párrafo completo sin detenerse abruptamente. 
            Instrucción: Escribe una mini-historia continua de 3 oraciones conectadas. Usa conectores simples como 'y', 'luego', 'después'. La historia debe tener un inicio, desarrollo y final muy rápido (ej. Un sapo que busca su hoja para dormir). Máximo 35 palabras en total.""",

            7: """Nivel 7 - Juego de Palabras (Rimas sin tensión): 
            Objetivo: Jugar con la fonética sin causar bloqueos. 
            Instrucción: Crea 2 mini-trabalenguas que sean MUY SUAVES. No deben ser difíciles de pronunciar, sino divertidos (juegos de aliteración con sonidos fáciles como la 'l' o la 's', ej. 'La luna ilumina la laguna'). Nada que fuerce demasiado la mandíbula o la lengua.""",

            8: """Nivel 8 - Voces Divertidas (Diálogos cortos): 
            Objetivo: Practicar cambios de entonación y toma de turnos. 
            Instrucción: Escribe un diálogo de 4 líneas entre dos personajes (ej. un león y un ratón). Usa obligatoriamente signos de interrogación (?) y exclamación (!) para que el niño practique cambiar el volumen y el tono de su voz. Indica qué personaje habla en cada línea.""",

            9: """Nivel 9 - Explicando el Mundo: 
            Objetivo: Lenguaje expositivo básico. 
            Instrucción: Escribe un texto explicativo de 4 oraciones donde se responda a una pregunta curiosa (ej. ¿Por qué brillan las estrellas? o ¿Cómo hacen miel las abejas?). El vocabulario puede ser un poquito más avanzado, ideal para niños de 8 a 10 años, pero manteniendo oraciones claras.""",

            10: """Nivel 10 - El Gran Cuentacuentos (Prosodia y Emoción): 
            Objetivo: Controlar la respiración bajo diferentes emociones. 
            Instrucción: Escribe un cuento corto de 5 oraciones. La historia debe cambiar de emoción explícitamente: empezar con misterio o tristeza, pasar a la sorpresa, y terminar con mucha alegría. Esto ayuda al niño a practicar su fluidez mientras maneja variaciones emocionales en su voz."""
        }
    else:
        if datos.perfil == "juvenil":
        contexto_rol = """
        ACTÚA COMO: Un mentor experto en comunicación juvenil y coach de debates para adolescentes.
        TONO: Empoderador, moderno, directo y empático. Debes sonar como un hermano mayor experto o un profesor joven y "cool". 
        TEMÁTICAS: Redes sociales (TikTok/YouTube), videojuegos, medio ambiente, justicia social, dilemas escolares, amistad, y futuro profesional.
        OBJETIVO: Ayudar al usuario a proyectar seguridad, reducir la ansiedad social y mejorar la fluidez mediante técnicas de comunicación asertiva.
        """
        
        niveles_dificultad = {
            0: """EVALUACIÓN DE CONFIANZA: 
            Genera un párrafo de 30 palabras donde el usuario se presenta a sí mismo en un nuevo grupo (ej: club de robótica o equipo de fútbol). 
            Objetivo: Medir velocidad del habla y seguridad inicial.""",

            1: """NIVEL 1 - EL BREAKING ICE (Inicios Suaves): 
            Genera 5 frases cortas para iniciar una conversación en el recreo o pasillo. 
            Regla Fonética: Deben iniciar con sonidos vocálicos o nasales suaves para evitar el bloqueo inicial. 
            Ejemplo: 'Oye, ¿has visto...', 'Me parece que...'""",

            2: """NIVEL 2 - OPINIÓN RÁPIDA (Sílabas conectadas): 
            Genera 4 frases de reacción de 5 palabras sobre un tema tendencia (ej: una nueva serie). 
            Instrucción: Enfócate en la unión de palabras (coarticulación) para que el pensamiento y el habla vayan al mismo ritmo.""",

            3: """NIVEL 3 - EL CHAT EN VOZ ALTA (Fluidez informal): 
            Genera un párrafo de 3 líneas simulando un audio de WhatsApp explicando un plan para el fin de semana. 
            Estilo: Coloquial pero estructurado, usando conectores naturales.""",

            4: """NIVEL 4 - RESPIRA Y OPINA (Pausas de control): 
            Genera un texto de 3 oraciones largas donde el adolescente da su punto de vista sobre el uso del uniforme escolar.
            Requisito: Coloca comas estratégicas para forzar pausas de inhalación antes de dar un argumento fuerte.""",

            5: """NIVEL 5 - EXPOSICIÓN DE CLASE (Lectura Informativa): 
            Genera un fragmento de 4 líneas sobre un dato curioso de tecnología o ciencia (ej: Inteligencia Artificial). 
            Objetivo: Mantener la fluidez al leer datos técnicos sin acelerarse por el nerviosismo.""",

            6: """NIVEL 6 - DEBATE EN EL AULA (Argumentación): 
            Genera un texto donde el usuario defiende una postura (ej: ¿Son mejores los libros físicos o digitales?). 
            Estructura: 'Yo opino que... porque... además...'. Ayuda a organizar el pensamiento lógico para reducir bloqueos.""",

            7: """NIVEL 7 - STORYTELLING SOCIAL (Narrativa fluida): 
            Genera un párrafo donde el usuario cuenta una anécdota graciosa que le pasó recientemente. 
            Objetivo: Mantener el hilo conductor y el entusiasmo sin perder el control de la respiración.""",

            8: """NIVEL 8 - DEFENDIENDO TU PUNTO (Manejo de presión): 
            Genera un diálogo donde un amigo está en desacuerdo con el usuario sobre un videojuego o deporte. 
            Instrucción: El usuario debe responder con asertividad y calma, manteniendo la fluidez a pesar de la "confrontación" simulada.""",

            9: """NIVEL 9 - MINI TED TALK (Discurso Público): 
            Genera un discurso motivador de 5 líneas para convencer a sus compañeros de reciclar o participar en un evento. 
            Enfoque: Uso de retórica, énfasis en palabras clave y contacto visual imaginario.""",

            10: """NIVEL 10 - EL GRAN RETO (Entrevista o Presentación Final): 
            Simula una situación de alta importancia: presentarse para una beca, una audición o una entrevista para su primer trabajo de verano. 
            Requisito: El texto debe incluir preguntas complejas que el usuario debe leer y responder con una estructura profesional y fluidez total."""
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
        logger.info(f"--- PROMPT ENVIADO A GEMINI ---\n{prompt}") # <--- AGREGA ESTA LÍNEA
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
            "texto": "Tenemos problemas en el servidor, intenta nuevamente en un minuto.", 
            "guia": "Parece que el búho esta dormido, danos un minuto para despertarlo."
        }

# ---------------------------------------------------------
# FASE 2: ANÁLISIS FINAL Y RECOMENDACIONES (NLP)
# ---------------------------------------------------------
@app.post("/analizar_ejercicio")
async def analizar_ejercicio(datos: PeticionAnalisis):
    logger.info(f"Petición recibida - Analizar Ejercicio. Original: '{datos.texto_original[:20]}...' Leído: '{datos.texto_leido[:20]}...'") # <--- AGREGADO

    model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"temperature": 0.85})
    
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
        "nivel_recommended": Solo si el nivel actual era 0, coloca aquí el número de nivel (1-10) sugerido. Si no, devuelve null.
    }}
    """
    
    try:
        logger.info(f"--- PROMPT ENVIADO A GEMINI ---\n{prompt_analisis}") # <--- AGREGA ESTA LÍNEA
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
            "analisis": "¡Vaya! El búho se tomó un pequeño descanso técnico.",
            "consejo": "Hubo un inconveniente con la conexión al cerebro de la IA. Por favor, reintenta en un momento.",
            "nivel_recommended": None
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
