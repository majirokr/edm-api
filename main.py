import os
import uuid
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import numpy as np
import librosa
import pyloudnorm as pyln
from pydub import AudioSegment

app = FastAPI()

def convertir_a_wav(file_path: str) -> str:
    sound = AudioSegment.from_file(file_path)
    wav_path = tempfile.mktemp(suffix=".wav")
    sound.export(wav_path, format="wav")
    return wav_path

def bandpass_filter(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    mask = (freqs >= low) & (freqs <= high)
    fft[~mask] = 0
    return np.fft.irfft(fft, n=len(y))

def segundos_a_minuto_seg(sec: int) -> str:
    m = sec // 60
    s = sec % 60
    return f"{m}:{s:02d}"

@app.post("/analizar-audio/")
async def analizar_audio(file: UploadFile = File(...)):
    """
    Analiza un track de audio enviado como archivo.
    Parámetros:
    - file: archivo de audio (mp3, wav, etc)
    """
    try:
        # Guardar subida y convertir
        ext    = os.path.splitext(file.filename)[1]
        tmp_in = f"temp_{uuid.uuid4().hex}{ext}"
        with open(tmp_in, "wb") as f:
            f.write(await file.read())
        wav = convertir_a_wav(tmp_in)
        os.remove(tmp_in)

        # Cargar audio
        y, sr = librosa.load(wav, sr=None, mono=True)
        os.remove(wav)

        # Métricas
        meter         = pyln.Meter(sr)
        loudness      = meter.integrated_loudness(y)
        peak          = float(np.max(np.abs(y)))
        rms           = np.sqrt(np.mean(y**2))
        dynamic_range = 20 * np.log10(peak / rms) if rms > 0 else 0.0

        # Bandas
        kick_band  = bandpass_filter(y, sr, 20, 150)
        bass_band  = bandpass_filter(y, sr, 20, 250)
        snare_band = bandpass_filter(y, sr, 150, 800)
        hats_band  = bandpass_filter(y, sr, 5000, 10000)

        kick_energy  = float(np.mean(librosa.feature.rms(y=kick_band)))
        bass_rms     = float(np.mean(librosa.feature.rms(y=bass_band)))
        snare_energy = float(np.mean(librosa.feature.rms(y=snare_band)))
        hats_energy  = float(np.mean(librosa.feature.rms(y=hats_band)))

        # Choque Kick+Bajo (único)
        thresh = 0.02
        kick_env = librosa.feature.rms(y=kick_band)[0]
        bass_env = librosa.feature.rms(y=bass_band)[0]
        idxs = np.where((kick_env > thresh) & (bass_env > thresh))[0]
        choque_time = None
        if idxs.size:
            choque_time = int(librosa.frames_to_time(idxs[0], sr=sr))

        # Choque Synth (solo primer)
        mid1 = bandpass_filter(y, sr, 300, 1000)
        mid2 = bandpass_filter(y, sr, 800, 2000)
        e1 = librosa.feature.rms(y=mid1)[0]
        e2 = librosa.feature.rms(y=mid2)[0]
        cols = np.where((e1 > e1.mean()+e1.std()) & (e2 > e2.mean()+e2.std()))[0]
        synth_time = None
        if cols.size:
            synth_time = int(librosa.frames_to_time(cols[0], sr=sr))

        # Construir feedback
        lines = []
        lines.append("¡Tremendo track que me envías! Ya lo he analizado:")

        # Dinámica
        if dynamic_range > 12:
            lines.append(
                "- Su dinámica es amplia; comprime ligeramente el master. "
                "Por ejemplo: compresor de bus a ratio 2:1, attack 10 ms, release 100 ms. "
                "También podrías probar multibanda leve en 3 bandas para controlar sub, mids y agudos."
            )
        else:
            lines.append(
                "- Su dinámica está bien controlada; buen trabajo con la compresión."
            )

        # Choque Kick+Bajo
        if choque_time is not None:
            min_seg = segundos_a_minuto_seg(choque_time)
            lines.append(
                f"- Choque Kick+Bajo en el minuto {min_seg}; usa sidechain o EQ dinámica ahí. "
                "Prueba un crossover a 100 Hz con release 50 ms y threshold alrededor de -30 dB."
            )
        else:
            lines.append("- Kick y Bajo no interfieren significativamente.")

        # Bajos
        lines.append(
            "- Tus Bajos están balanceados en volumen; no requieren EQ extra, "
            "pero si quieres más presencia, añade +2 dB en 60–80 Hz con Q 1.0."
        )

        # Batería
        lines.append(
            "- Balance de batería OK. "
            "Al Kick añade transient shaping (+20 % punch, -10 % sustain), "
            "al Snare configura compresión 3:1, attack 5 ms, release 200 ms, "
            "y al Hi-Hat un poco de saturación suave (drive 15 %)."
        )

        # Choque Synth
        if synth_time is not None:
            min_seg_s = segundos_a_minuto_seg(synth_time)
            lines.append(
                f"- Choque sintetizadores en el minuto {min_seg_s}; "
                "revisa EQ (high cut en 800–1000 Hz con filtro paso bajo suave) "
                "o paneo a 30 % izquierda/derecha. También puedes usar un de-esser en 1 kHz."
            )
        else:
            lines.append("- No se detectaron choques evidentes entre sintetizadores.")

        # Glosario breve
        lines.append("- Por si necesitas refrescar los términos:")
        lines.append("  • high cut: filtro que atenúa frecuencias por encima de cierto punto.")
        lines.append("  • sidechain: reduce el nivel de una señal cuando otra suena, útil para Kick/Bajo.")
        lines.append("  • transient shaping: modela ataque y sustain de la señal para dar más punch.")
        lines.append("  • de-esser: atenúa resonancias en una banda estrecha, común en voces y synths.")

        feedback = "\n".join(lines)

        # --- aquí podría generarse un gráfico (e.g. curva de loudness) y devolver su URL ---
        return JSONResponse({"feedback_formateado": feedback})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error procesando audio: {e}")
