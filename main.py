from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Esto define qué datos nos enviará tu App de Flutter
class DatosUsuario(BaseModel):
    perfil: str  # Puede ser "infantil" o "adulto"
    nivel: int

@app.get("/")
def inicio():
    return {"mensaje": "¡Cerebro de NeuroRead encendido!"}

@app.post("/generar_ejercicio")
async def generar(datos: DatosUsuario):
    # Lógica de simulación para niño vs adulto
    if datos.perfil == "infantil":
        return {
            "texto": "El pato Pepe patina en el parque.",
            "guia": f"Nivel {datos.nivel}: Enfócate en la letra P",
            "tipo": "Niño"
        }
    else:
        return {
            "texto": "La fluidez lectora mejora la comprensión cognitiva.",
            "guia": f"Nivel {datos.nivel}: Practica la entonación",
            "tipo": "Adulto"
        }