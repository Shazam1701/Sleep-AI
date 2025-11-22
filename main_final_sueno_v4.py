import os
import re
from io import BytesIO

import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tensorflow.keras.models import load_model
import joblib

from prompts_sueno import stronger_prompt_sueno

# ============================================
# CONFIGURACIÓN INICIAL (API + MODELO)
# Carga claves de OpenAI, 
# inicializa modelos Whisper / GPT-4o y carga }
# el modelo ANN, scaler y label encoder.
# ============================================

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY. Verifica tu archivo .env.")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

# Modelos OpenAI
MODEL_CHAT = "gpt-4o-mini"        # Chat principal
MODEL_TRANSCRIBE = "whisper-1"    # Voz → Texto
MODEL_TTS = "gpt-4o-mini-tts"     # Texto → Voz


@st.cache_resource
def load_artifacts():
    try:
        # Cargar modelo Keras
        model = load_model("modelos/modelo_sleep.keras")

        # Cargar scaler y encoder
        scaler = joblib.load("modelos/scaler_sleep.pkl")
        label_encoder = joblib.load("modelos/label_encoder_sleep.pkl")

        return model, scaler, label_encoder

    except Exception as e:
        st.error(f"❌ Error cargando artefactos: {e}")
        raise e


model_ann, scaler_sleep, label_encoder_sleep = load_artifacts()



# ============================================
# ❓ PREGUNTAS DEL FLUJO GUIADO
# ============================================

PREGUNTAS = [
    ("Age", "Para comenzar, ¿cuántos años tienes? "),
    ("Sleep Duration", "¿Cuántas horas duermes normalmente al día? (Por ejemplo: 6.5)"),
    ("Stress Level", "En una escala del 0 al 10 donde 0 es nada y 10 mucho, ¿qué tan estresado te encuentras?"),
    ("Physical Activity Level", "¿Cuál es el promedio de minutos de actividad física en tu día?"),
]

FEATURE_ORDER = [key for key, _ in PREGUNTAS]


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def extraer_numero(texto, tipo="float"):
    """
    Extrae el primer número del texto.
    tipo: "int" o "float"
    """
    if texto is None:
        return None
    coincidencias = re.findall(r"[-+]?\d*\.?\d+", texto.replace(",", "."))
    if not coincidencias:
        return None
    valor = coincidencias[0]
    try:
        if tipo == "int":
            return int(float(valor))
        else:
            return float(valor)
    except ValueError:
        return None


def predecir_calidad_sueno(input_dict):
    """
    Usa el modelo ANN para estimar la calidad del sueño.

    Versión robusta:
    - Respeta FEATURE_ORDER
    - Maneja errores del modelo
    - Devuelve clase y diccionario de probabilidades por etiqueta
    """
    try:
        df = pd.DataFrame([input_dict])[FEATURE_ORDER]
        X = scaler_sleep.transform(df)
        proba = model_ann.predict(X)[0]
        clase_idx = int(np.argmax(proba))
        clase = label_encoder_sleep.inverse_transform([clase_idx])[0]
    except Exception as e:
        print("ERROR en predecir_calidad_sueno:", e)
        return "Unknown", {}

    # Validación básica de clase
    clases_validas = list(label_encoder_sleep.classes_)
    if clase not in clases_validas:
        return "Unknown", {}

    proba_dict = {
        label_encoder_sleep.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(proba)
    }

    return clase, proba_dict


def generar_audio(texto):
    """
    Genera audio en MP3 a partir de un texto usando TTS.

    """
    try:
        speech = client_openai.audio.speech.create(
            model=MODEL_TTS,
            voice="alloy",
            input=texto
        )
        audio_bytes = speech.read()
        return audio_bytes
    except Exception as exc:
        st.error(f"No se pudo generar audio: {exc}")
        return None


# ============================================
# REPORTE EJECUTIVO
# ============================================

def generar_reporte_ejecutivo(inputs):
    edad = inputs.get("Age")
    duracion = inputs.get("Sleep Duration")
    estres = inputs.get("Stress Level")
    actividad = inputs.get("Physical Activity Level")

    recomendaciones = []

    # Edad
    if edad is not None:
        if edad < 25:
            recomendaciones.append(
                "A tu edad, el cuerpo requiere entre 7–9 horas de sueño para optimizar aprendizaje y recuperación."
            )
        elif edad < 40:
            recomendaciones.append(
                "Entre los 25 y 40 años, mantener un sueño regular reduce el riesgo de estrés crónico."
            )
        else:
            recomendaciones.append(
                "A partir de los 40, la calidad del sueño tiende a disminuir; prioriza horarios consistentes y buena higiene del sueño."
            )

    # Duración del sueño
    if duracion is not None:
        if duracion < 6:
            recomendaciones.append(
                "Duermes menos de 6 horas. Esto eleva estrés, apetito y fatiga. Intenta acercarte a 7–8 horas de sueño real."
            )
        elif duracion < 7:
            recomendaciones.append(
                "Tu sueño está cerca del nivel óptimo, pero podrías beneficiarte de alcanzar 7–8 horas constantes."
            )
        else:
            recomendaciones.append(
                "Tu duración de sueño es adecuada. Mantén horarios regulares y evita pantallas antes de dormir."
            )

    # Estrés
    if estres is not None:
        if estres >= 7:
            recomendaciones.append(
                "Tu nivel de estrés es alto. Considera pausas activas, respiración profunda o pequeños descansos durante el día."
            )
        elif estres >= 4:
            recomendaciones.append(
                "Tu nivel de estrés es moderado. Mantener una rutina estable de sueño ayudará a evitar que aumente."
            )
        else:
            recomendaciones.append(
                "Tu nivel de estrés es bajo, lo cual favorece un sueño más profundo y reparador. ¡Sigue así!"
            )


    # Actividad física
    if actividad is not None:
        if actividad < 30:
            recomendaciones.append(
                "Tu actividad física es baja. Caminar a paso rapido o trotar al menos 30 minutos al día puede mejorar significativamente tu calidad de sueño."
            )
        elif actividad < 60:
            recomendaciones.append(
                "Tu actividad física es moderada. Mantenerla o incrementarla ligeramente puede favorecer aún más tu descanso."
            )
        else:
            recomendaciones.append(
                "Tienes un excelente nivel de actividad física, lo que favorece un mejor ciclo sueño-vigilia."
            )

    if not recomendaciones:
        recomendaciones.append(
            "No se pudieron generar recomendaciones específicas. Verifica que hayas respondido todas las preguntas."
        )

    reporte = "\n".join([f"- {r}" for r in recomendaciones])
    return reporte


# ============================================
# COMANDOS ESPECIALES (CANCELAR / REINICIAR)
# ============================================

def detectar_comando_especial(texto):
    texto = texto.lower().strip()

    comandos_cancelar = [
        "cancelar", "cancel", "stop", "detener", "salir",
        "ya no quiero", "ya no", "no quiero seguir"
    ]

    comandos_reiniciar = [
        "reiniciar", "restart", "volver a empezar", "empezar de nuevo",
        "reset", "desde cero"
    ]

    for c in comandos_cancelar:
        if c in texto:
            return "cancelar"

    for c in comandos_reiniciar:
        if c in texto:
            return "reiniciar"

    return None


def validar_respuesta_numerica(texto, key):
    """
    Valida que el usuario haya escrito un número.
    Usa el `key` de la variable para aplicar validaciones básicas de rango.
    Regresa: (valor_float, None) si es válido
             (None, mensaje_error) si no es válido
    """
    if texto is None:
        return None, "Necesito un número numérico (por ejemplo: 25 o 7.5). ¿Puedes repetirlo?"

    # Extraer número desde el texto (por si dice "tengo 25 años")
    numeros = re.findall(r"[-+]?\d*\.\d+|\d+", texto)

    if not numeros:
        return None, "Necesito un número numérico (por ejemplo: 25 o 7.5). ¿Puedes repetirlo?"

    try:
        valor = float(numeros[0])
    except ValueError:
        return None, "Lo que escribiste no parece un número válido. Intenta solo con números."

    # Validaciones sencillas por variable
    if key == "Age" and not (0 < valor < 120):
        return None, "La edad debe estar entre 1 y 120 años."
    if key == "Sleep Duration" and not (0 < valor <= 24):
        return None, "Las horas de sueño deben estar entre 0 y 24."
    if key == "Physical Activity Level" and valor < 0:
        return None, "Los minutos de actividad física no pueden ser negativos."
    if key == "Stress Level" and not (0 <= valor <= 10):
        return None, "El nivel de estrés debe estar entre 0 y 10."

    return valor, None


# ============================================
# ESTADO INICIAL (SESSION STATE)
# ============================================

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hola! Soy SleepIA. "
                "Podemos platicar de tus hábitos de sueño o, si lo prefieres, "
                "puedo hacerte unas preguntas para analizar tu calidad de descanso con un modelo de Inteligencia Artificial."
            ),
        }
    ]

if "modo_analisis" not in st.session_state:
    st.session_state["modo_analisis"] = False

if "inputs_usuario" not in st.session_state:
    st.session_state["inputs_usuario"] = {}

if "indice_pregunta" not in st.session_state:
    st.session_state["indice_pregunta"] = 0


# ============================================
# ONFIGURACIÓN DE INTERFAZ 
# ============================================

st.set_page_config(page_title="SleepIA", page_icon="🌙 😴 💤")
st.title("💤 Sleep AI")
st.caption("🌙 Soy un Chat LLM con un Modelo de Red Neuronal Artificial que clasifica la calidad del sueño promedio")

# Contenedor de chat
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        message_block = st.chat_message(msg["role"])
        message_block.write(msg["content"])
        audio_payload = msg.get("audio")
        if audio_payload:
            message_block.audio(audio_payload, format="audio/mp3")


# ============================================
# SIDEBAR: CONTROLES
# ============================================

with st.sidebar:
    st.subheader("🎧 Entrada por voz")
    
    st.markdown("""
    **Cómo usar la entrada por voz:**
    1. Presiona el microfono para Grabar audio.
    2. Habla con normalidad (máximo 15–20 segundos).
    3. Da clic en el boton de Detener (De color rojo)
    4. Da clic en **Enviar audio** para transcribirlo.
    
    > Tip: Puedes responder preguntas del modelo con esta funcion.
    """)
    audio_input = st.audio_input("Graba un mensaje de voz (opcional)")
    send_audio = st.button("Enviar audio", use_container_width=True)

    st.markdown("---")
    st.subheader("🧪 Análisis con modelo ANN")
    iniciar_analisis = st.button(
        "Iniciar preguntas del modelo",
        use_container_width=True
    )

# Si el usuario inicia el análisis guiado
if iniciar_analisis:
    if not st.session_state.get("modo_analisis", False):

        # Activar modo análisis
        st.session_state["modo_analisis"] = True
        st.session_state["inputs_usuario"] = {}
        st.session_state["indice_pregunta"] = 0

        # Forzar un mensaje de usuario "fantasma" para refrescar el chat
        st.session_state.messages.append({
            "role": "user",
            "content": " "
        })

        # Mostrar primer mensaje del flujo guiado
        key, pregunta_texto = PREGUNTAS[0]
        texto = (
            "Perfecto 😴💡\n\n"
            "Vamos a comenzar tu evaluación del sueño con IA.\n\n"
            f"{pregunta_texto}"
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": texto
        })

        st.rerun()

# ============================================
# ENTRADA DE TEXTO / AUDIO
# ============================================

user_prompt = None
user_display = None

# Texto tiene prioridad
if text_prompt := st.chat_input(
    "Escribe cómo dormiste o responde a la pregunta del modelo..."
):
    user_prompt = text_prompt
    user_display = text_prompt

# Si no hay texto, puede venir audio
elif send_audio:
    if audio_input is not None:
        raw_audio = audio_input.getvalue()
        filename = audio_input.name or "voz_usuario.wav"
        audio_file = BytesIO(raw_audio)
        audio_file.name = filename

        with st.spinner("🎧 Transcribiendo tu mensaje..."):
            transcription = client_openai.audio.transcriptions.create(
                model=MODEL_TRANSCRIBE,
                file=audio_file,
            )
        user_prompt = transcription.text.strip()
        user_display = f"(Audio) {user_prompt}" if user_prompt else None

        if not user_prompt:
            st.warning("⚠️ No se detectó texto en la grabación. Intenta nuevamente.")
    else:
        st.warning("Por favor graba un audio antes de enviarlo.")


# ============================================
# LÓGICA PRINCIPAL: CHAT + FLUJO ANN (VERSIÓN B)
# ============================================

def manejar_respuesta_analisis(user_text: str):
    """
    Flujo guiado versión B (v3) adaptado al diseño original.
    - Soporta comandos cancelar/reiniciar
    - Valida número por variable
    - Al final llama al ANN, genera reporte y audio
    """

    # =====================================================================
    # SI YA SE ACTIVÓ EL PROCESAMIENTO, GENERAMOS LA PREDICCIÓN DIRECTO
    # =====================================================================
    if st.session_state.get("procesando_resultado", False):

        inputs = st.session_state["inputs_usuario"].copy()

        with st.spinner("🧠 Analizando tu patrón de sueño..."):
            clase, proba = predecir_calidad_sueno(inputs)

        recomendaciones_por_clase = {
            "Excelente": [
                "Mantén una rutina de sueño consistente.",
                "Evita pantallas al menos 45 minutos antes de dormir.",
                "Procura mantener tus buenos hábitos de descanso."
            ],
            "Buena": [
                "Intenta dormir entre 7 y 8 horas reales.",
                "Reduce la cafeína después de las 4 PM.",
                "Establece horarios más constantes para acostarte."
            ],
            "Regular": [
                "Tu descanso podría mejorar significativamente.",
                "Mejora tu higiene del sueño (luz, ruido, temperatura).",
                "Considera técnicas de manejo de estrés o hablar con un especialista si persiste."
            ],
        }

        rec = recomendaciones_por_clase.get(clase, ["Mejora tus hábitos de sueño."])
        while len(rec) < 3:
            rec.append("Continúa mejorando tus hábitos para un mejor descanso.")

        reporte = generar_reporte_ejecutivo(inputs)

        txt_final = f"""
😴 **Resultados de tu evaluación del sueño**

📌 Calidad estimada de tu sueño: **{clase}**

💡 **Recomendaciones personalizadas:**
- {rec[0]}
- {rec[1]}
- {rec[2]}

📘 **Reporte Ejecutivo Personalizado**
{reporte}

Si deseas otra evaluación, puedes indicarlo cuando quieras 🤍
"""

        audio = generar_audio(txt_final)

        st.session_state.messages.append({
            "role": "assistant",
            "content": txt_final,
            "audio": audio,
        })

        # RESET DEL FLUJO
        st.session_state["procesando_resultado"] = False
        st.session_state["modo_analisis"] = False
        st.session_state["indice_pregunta"] = 0
        st.session_state["inputs_usuario"] = {}

        st.rerun()
        return

    # =====================================================================
    # COMANDOS ESPECIALES
    # =====================================================================
    comando = detectar_comando_especial(user_text or "")
    if comando == "cancelar":
        st.session_state["modo_analisis"] = False
        st.session_state["indice_pregunta"] = 0
        st.session_state["inputs_usuario"] = {}
        st.session_state.messages.append({
            "role": "assistant",
            "content": "He cancelado la evaluación del sueño. Podemos seguir platicando de forma libre 😊."
        })
        return

    elif comando == "reiniciar":
        st.session_state["modo_analisis"] = True
        st.session_state["indice_pregunta"] = 0
        st.session_state["inputs_usuario"] = {}
        _, txt = PREGUNTAS[0]
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Reiniciamos la evaluación desde el inicio.\n\n" + txt
        })
        return

    # =====================================================================
    # PARCHE ANTI-INDEXERROR
    # =====================================================================
    idx = st.session_state.get("indice_pregunta", 0)

    if not isinstance(idx, int) or idx < 0 or idx >= len(PREGUNTAS):
        st.session_state["modo_analisis"] = True
        st.session_state["indice_pregunta"] = 0
        st.session_state["inputs_usuario"] = {}
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Hubo un pequeño desajuste en el orden de las preguntas 😅.\n"
                "Vamos a reiniciar la evaluación desde el inicio.\n\n"
                f"{PREGUNTAS[0][1]}"
            )
        })
        return

    # =====================================================================
    # VALIDACIÓN DE RESPUESTA NUMÉRICA
    # =====================================================================
    key, _ = PREGUNTAS[idx]
    valor, error_msg = validar_respuesta_numerica(user_text, key)

    if error_msg:
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.rerun()
        return

    # =====================================================================
    # GUARDAR RESPUESTA
    # =====================================================================
    st.session_state["inputs_usuario"][key] = valor
    st.session_state["indice_pregunta"] += 1

    # =====================================================================
    # ¿AÚN HAY PREGUNTAS?
    # =====================================================================
    if st.session_state["indice_pregunta"] < len(PREGUNTAS):
        _, siguiente_txt = PREGUNTAS[st.session_state["indice_pregunta"]]
        st.session_state.messages.append({
            "role": "assistant",
            "content": siguiente_txt
        })
        st.rerun()
        return

    # =====================================================================
    # 🏁 FIN DEL FLUJO → ACTIVAR PREDICCIÓN ANN
    # =====================================================================
    st.session_state["procesando_resultado"] = True

    st.session_state.messages.append({
        "role": "assistant",
        "content": "⏳ Procesando tu información... dame unos segundos 😴🌙"
    })

    st.rerun()

    # Reset del flujo
    st.session_state["modo_analisis"] = False
    st.session_state["indice_pregunta"] = 0
    st.session_state["inputs_usuario"] = {}

# ============================================
# 🚀 Procesamiento automático tras el mensaje de carga
# ============================================

if st.session_state.get("procesando_resultado", False):
    manejar_respuesta_analisis("")   # ejecuta siguiente paso sin requerir prompt


if user_prompt:
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_display or user_prompt)

    # Si estamos en modo análisis → usamos flujo ANN robusto
    if st.session_state["modo_analisis"]:
        manejar_respuesta_analisis(user_prompt)
    else:
        # Conversación libre con el modelo de lenguaje
        conversation = [{"role": "system", "content": stronger_prompt_sueno}]
        conversation.extend(
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        )

        with st.chat_message("assistant"):
            with st.spinner("Analizando tus patrones de sueño... 😴"):
                stream = client_openai.chat.completions.create(
                    model=MODEL_CHAT,
                    messages=conversation,
                    stream=True
                )
                respuesta = st.write_stream(stream)

        # Guardar respuesta como texto
        nuevo_msg = {"role": "assistant", "content": respuesta}

        # Generar audio de la respuesta (como en el diseño original)
        audio_bytes = generar_audio(respuesta)
        if audio_bytes:
            nuevo_msg["audio"] = audio_bytes

        st.session_state.messages.append(nuevo_msg)
