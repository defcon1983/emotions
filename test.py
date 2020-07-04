import emotion_rec  # se importa el modulo completo
import pathlib
from pathlib import Path  # se importa el modulo completo

"""
IMPORTANTE: ES VITAL QUE EL MODULO (emotion_rec) ESTE EN EL MISMO DIRECTORIO DEL SCRIPT QUE LO 
USA.

Este modulo muestra el uso basico de la funcion emotion_rec
"""

root = Path(".")
p2img = root / "cara.jpg"

# el argumento de la funcion es un string que contenga el path de la imagen que se va a alimentar
emotions = emotion_rec.emotion_detect(str(p2img))

# la funcion regresa un array de las emociones detectadas en la imagen
print(emotions)

