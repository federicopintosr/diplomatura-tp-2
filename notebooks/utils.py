"""Funciones compartidas del TP2.

La limpieza vive aca (y no dentro de una notebook) para que la notebook 03
(train) y la 06 (entrega) apliquen EXACTAMENTE la misma transformacion.
Todo es fila por fila: no se calcula ningun estadistico del conjunto, asi que
no hay fuga de informacion aunque se aplique antes del split.
"""
import numpy as np
import pandas as pd

TARGET = "smoking"
ID_COL = "id"

# Nombres originales -> snake_case (sin parentesis ni espacios)
RENOMBRAR = {
    "ID": "id", "gender": "gender", "age": "age",
    "height(cm)": "height_cm", "weight(kg)": "weight_kg", "waist(cm)": "waist_cm",
    "eyesight(left)": "eyesight_left", "eyesight(right)": "eyesight_right",
    "hearing(left)": "hearing_left", "hearing(right)": "hearing_right",
    "systolic": "systolic", "relaxation": "relaxation",
    "fasting blood sugar": "fasting_blood_sugar", "Cholesterol": "cholesterol",
    "triglyceride": "triglyceride", "HDL": "hdl", "LDL": "ldl",
    "hemoglobin": "hemoglobin", "Urine protein": "urine_protein",
    "serum creatinine": "serum_creatinine", "AST": "ast", "ALT": "alt", "Gtp": "gtp",
    "oral": "oral", "dental caries": "dental_caries", "tartar": "tartar",
    "smoking": "smoking",
}

DESCARTAR = ["oral"]  # constante (vale "Y" en el 100% de ambos archivos)

# Columnas que Excel guardo como hora (ver notebook 03, seccion 2)
COLS_HORA = [
    "waist_cm", "eyesight_left", "eyesight_right", "hearing_left", "hearing_right",
    "systolic", "relaxation", "fasting_blood_sugar", "cholesterol", "triglyceride",
    "hdl", "ldl", "hemoglobin", "urine_protein", "serum_creatinine", "ast", "alt", "gtp",
]
COLS_ENTERAS = ["hearing_left", "hearing_right", "urine_protein"]  # quedan como int

CODIGO_CEGUERA = 9.9  # en eyesight no es una agudeza visual, es un codigo

BINARIAS = {"gender": {"M": 1, "F": 0}, "tartar": {"Y": 1, "N": 0}}


def deshacer_hora_excel(s):
    """Revierte un numero que Excel leyo como hora h.mm.

    Excel guardo `1.2` como la hora 1:02, o sea 1 + 2/60 horas = 0.0430 dias.
    Se pasa a minutos (x * 24 * 60), y se arma de nuevo horas + minutos / 10.
    """
    minutos = (s * 24 * 60).round()
    if ((s * 24 * 60 - minutos).abs() > 1e-6).any():
        raise ValueError(f"{s.name}: hay valores que no son minutos exactos")
    if (minutos % 60 > 9).any():
        raise ValueError(f"{s.name}: hay 'minutos' > 9, no es un decimal de una cifra")
    return minutos // 60 + (minutos % 60) / 10


def grupos_duplicados(df, columnas):
    """Numero de grupo por fila: filas con valores identicos en `columnas` comparten grupo.

    El dataset tiene registros duplicados de la misma persona con distinto ID. Usar
    estos grupos al separar datos evita que una copia quede en train y otra en evaluacion.
    """
    return df.groupby(columnas, sort=False).ngroup().to_numpy()


def preparar(df):
    """Limpieza completa: misma funcion para train y para entrega."""
    df = df.rename(columns=RENOMBRAR).drop(columns=DESCARTAR)

    if df.isna().any().any():
        raise ValueError("Hay nulos: revisar el archivo antes de seguir")

    for c in COLS_HORA:
        df[c] = deshacer_hora_excel(df[c])
    df[COLS_ENTERAS] = df[COLS_ENTERAS].astype(int)

    for c in ["eyesight_left", "eyesight_right"]:
        df[c] = df[c].replace(CODIGO_CEGUERA, 0.0)

    for c, mapa in BINARIAS.items():
        desconocidos = set(df[c]) - set(mapa)
        if desconocidos:
            raise ValueError(f"{c}: valores no vistos {desconocidos}")
        df[c] = df[c].map(mapa).astype(int)

    return df
