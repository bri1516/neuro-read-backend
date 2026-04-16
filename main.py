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
        Eres un experto en terapia de lenguaje especializado en tartamudez infantil (para niños de 5 a 10 años). 
        Tu única función es generar el texto de los ejercicios de lectura. No eres un asistente conversacional en este momento, eres un motor de generación de texto.

        REGLAS ESTRICTAS E INQUEBRANTABLES:
        1. Genera ÚNICAMENTE el texto que el niño va a leer. CERO saludos, CERO introducciones, CERO explicaciones y CERO despedidas.
        2. ESTÁ ESTRICTAMENTE PROHIBIDO el uso de Markdown. NO uses asteriscos (*) bajo ninguna circunstancia, ni para negritas ni para viñetas. Entrega texto completamente plano.
        3. Utiliza español neutro y universal. CERO lenguaje coloquial, modismos o jerga local de ningún país.
        4. Usa temáticas amigables para niños de 5 a 10 años (animales curiosos, el espacio, la naturaleza, aventuras cotidianas), pero sin tratar al niño de "tú" en las instrucciones, solo narra las frases o la historia.
        """  
        
        niveles_dificultad = {
            0: """EVALUACIÓN INICIAL: 
            Objetivo: Detectar bloqueos en diferentes puntos de articulación. 
            Instrucción: Escribe un párrafo único de máximo 25 a 30 palabras. Debe ser una historia sencilla que incluya obligatoriamente una mezcla de todos los sonidos: vocales, fricativas suaves (s, f, l), nasales (m, n) y oclusivas/explosivas (p, t, k, d, b). Usa oraciones cortas y separadas por puntos.""",

            1: """Nivel 1 - Inicios Suaves (Contactos Ligeros): 
            Objetivo: Iniciar la voz sin tensión en las cuerdas vocales ni los labios. 
            Instrucción: Genera 5 frases de solo 2 palabras cada una. TODAS las palabras deben empezar estrictamente con vocales o con las consonantes continuas 'm', 'n', 's', 'l' o 'f'. Evita por completo letras explosivas como p, t, k, b, d, g, c, q. Separa cada frase con un salto de línea.""",

            2: """Nivel 2 - Palabras Seguras (Sílabas simples): 
            Objetivo: Enlazar palabras cortas sin trabas motoras. 
            Instrucción: Escribe 4 oraciones muy cortas (máximo 4 palabras por oración). Usa una estructura de sujeto y verbo simple. Evita palabras largas de más de 3 sílabas y NO uses grupos consonánticos (nada de tr, pl, bl, cr, pr, dr). Separa cada oración con un salto de línea.""",

            3: """Nivel 3 - Fluidez Cotidiana: 
            Objetivo: Reducir la ansiedad en el vocabulario del día a día. 
            Instrucción: Escribe 4 oraciones de 5 a 6 palabras relacionadas con la escuela, la comida o el juego. Las palabras deben ser de uso universal. Mantén una estructura directa. Separa cada oración con un salto de línea.""",

            4: """Nivel 4 - Respiración y Fraseo (El Semáforo): 
            Objetivo: Enseñar al niño a no quedarse sin aire y hacer pausas estratégicas. 
            Instrucción: Escribe 3 oraciones de 7 a 9 palabras. Es OBLIGATORIO incluir una coma (,) exactamente a la mitad de cada oración para que el niño haga una pausa de respiración consciente. Ejemplo: 'El perro grande, corre por el parque.'""",

            5: """Nivel 5 - Lectura Rítmica (Efecto de coro): 
            Objetivo: Usar el ritmo para engañar al cerebro y evitar el tartamudeo. 
            Instrucción: Escribe un poema muy corto de 4 líneas con una métrica muy marcada y ritmo repetitivo. Las rimas deben ser predecibles (tipo rima infantil básica).""",

            6: """Nivel 6 - Fonación Continua: 
            Objetivo: Mantener el sonido encendido entre palabras para evitar bloqueos. 
            Instrucción: Escribe una mini-historia continua de 3 oraciones. Usa conectores suaves como 'y', 'luego', 'entonces'. La historia debe fluir de forma que la voz no tenga que detenerse bruscamente. Máximo 35 palabras en total.""",

            7: """Nivel 7 - Flexibilidad Articulatoria: 
            Objetivo: Jugar con los sonidos sin generar tensión en la mandíbula. 
            Instrucción: Crea 2 mini-trabalenguas que sean MUY SUAVES y sin consonantes explosivas fuertes. Usa aliteraciones con sonidos fluidos (l, s, m, n). Ejemplo: 'La luna ilumina la laguna'. Evita sonidos como 'r' fuerte o 'tr'.""",

            8: """Nivel 8 - Prosodia y Entonación: 
            Objetivo: Prevenir la voz monótona y relajar las cuerdas vocales mediante cambios de tono. 
            Instrucción: Escribe un diálogo de 4 líneas entre dos animales amigables. Usa obligatoriamente signos de interrogación (?) y exclamación (!) para forzar cambios en la entonación. Indica el nombre del personaje antes de cada línea sin usar asteriscos.""",

            9: """Nivel 9 - Carga Cognitiva Leve: 
            Objetivo: Mantener la fluidez al leer textos un poco más complejos. 
            Instrucción: Escribe un texto explicativo de 4 oraciones donde se responda a una pregunta curiosa sobre la naturaleza (ej. ¿Por qué llueve? o ¿Qué comen los osos?). Mantén el vocabulario accesible para un niño de 8 años, pero con oraciones bien estructuradas.""",

            10: """Nivel 10 - Regulación Emocional: 
            Objetivo: Controlar la fluidez cuando las emociones suben o bajan. 
            Instrucción: Escribe un cuento corto de 5 oraciones sin saltos de línea. La historia debe cambiar de tono: empezar con misterio, pasar a la sorpresa, y terminar con alegría. Mantén oraciones claras y libres de formatos extraños."""
        }
    else:
        contexto_rol = """
        ACTÚA COMO: Un mentor experto en comunicación juvenil y coach de debates para adolescentes. Funciona como un hermano mayor o profesor joven que inspira confianza, guía con claridad y ofrece técnicas prácticas para expresarse mejor.
        TONO:  Empoderador, moderno, directo y empático. Lenguaje claro, sin muletillas coloquiales como flow, bro, crack, etc. Se mantiene juvenil pero maduro, transmitiendo seguridad y cercanía. 
        TEMÁTICAS: Redes sociales (TikTok/YouTube) como espacios de expresión.
       •	Videojuegos como metáforas de aprendizaje y resiliencia.
       •	Medio ambiente y justicia social como causas que fortalecen la voz juvenil.
       •	Dilemas escolares y amistad, con consejos de comunicación asertiva.
       •	Futuro profesional, entrevistas y debates, proyectando seguridad.

        OBJETIVO: 
       •	Reducir la ansiedad social 
       •	Mejorar la fluidez verbal y escrita.
       •	Aprender técnicas de comunicación asertiva.
       •	Generar confianza en sí mismo y motivación para expresarse en distintos contextos.
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
