"""Genera informe/informe.pdf (máx. 5 páginas + referencias) a partir de los archivos de evidencias/.

Todas las cifras salen de evidencias/ (auditoría, experimentos, selección, métricas de prueba, recuperación),
así el informe no puede discrepar de ellos. Uso (entorno con informe/requirements.txt):
    python informe/generar_informe.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as lienzo
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

E = Path("evidencias")
SALIDA = Path("informe/informe.pdf")
FIG = Path("informe/umbral_v2.png")
MARGEN = 1.7 * cm
BANDA = 5.1 * cm
# Paleta tomada de la plantilla del taller (colores de texto, rellenos y trazos del PDF del docente)
TINTA, TINTA2 = "#10251b", "#475467"          # texto principal y secundario
VERDE, ACENTO, DORADO = "#123f2b", "#1b5e3b", "#d7b85a"  # títulos, trazos, línea de acento
FONDO, TENUE, REGLA = "#10251b", "#f0f4f1", "#dde6df"    # banda superior, relleno de tablas, líneas
AZUL, ROJO = "#0072B2", "#D55E00"  # colores de los datos del gráfico: Okabe-Ito validado (CVD y contraste)
FUENTES = Path(__file__).parent / "fuentes"
TALLER_PDF = Path("Defiende-el-servidor-taller-evaluable-2026-10-16.pdf")  # solo local (ignorado por Git)
LOGO = Path("informe/logo_externado.png")  # se extrae del PDF del taller al generar; no se versiona


def n(x, d=0):
    """Número con coma decimal y punto de miles."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    return n(100 * x, d) + " %"


aud = json.load(open(E / "auditoria.json"))
exp = pd.read_csv(E / "experimentos.csv").set_index("config_id")
sel = json.load(open(E / "seleccion.json"))
tst = json.load(open(E / "metricas-test.json"))
rec = json.load(open(E / "recuperacion/resultado.json"))
concl = json.load(open(E / "conclusiones_validacion.json"))
mlf = {cid: json.load(open(E / f"mlflow/{rid}.json")) for cid, rid in exp.run_id.items()}
tr, te = aud["archivos"]["training"], aud["archivos"]["testing"]
v1, v2 = sel["v1"], sel["v2"]
m1, m2, mt = v1["metricas_validacion"], v2["metricas_validacion"], tst["metricas"]
p95_c2 = mlf["C2"]["metricas"]  # C2 en su propio umbral con recall mínimo 0,95 (referencia)


def figura():
    v = pd.read_csv("data/processed/validacion_taller-C4.csv")
    y, s = v.y_true.values, v.score_attack.values
    u = np.arange(0.30, 0.951, 0.005)
    rec_ = np.array([(s[y == 1] >= t).mean() for t in u])
    pre = np.array([(y[s >= t] == 1).mean() for t in u])
    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=220)
    ax.plot(u, pre, color=AZUL, lw=1.6)
    ax.plot(u, rec_, color=ROJO, lw=1.6)
    ax.axhline(0.95, color=TINTA2, lw=0.8, ls=(0, (4, 3)))
    for x, t in ((0.5, "v1: 0,5"), (0.66, "v2: 0,66")):
        ax.axvline(x, color=TINTA2, lw=0.8)
        ax.text(x + 0.004, 0.865, t, color=TINTA, fontsize=7.5, va="bottom")
    ax.text(0.952, pre[-1] + 0.004, "precisión", color=TINTA, fontsize=8, va="bottom", ha="right")
    ax.plot([0.952], [pre[-1]], "o", color=AZUL, ms=3)
    ax.text(0.952, rec_[-1] - 0.006, "recall", color=TINTA, fontsize=8, va="top", ha="right")
    ax.plot([0.952], [rec_[-1]], "o", color=ROJO, ms=3)
    ax.text(0.302, 0.953, "piso de recall 0,95", color=TINTA2, fontsize=7.5, va="bottom")
    ax.set_xlim(0.30, 0.96)
    ax.set_ylim(0.86, 1.0)
    ax.set_xlabel("umbral de decisión (validación, modelo de v2)", color=TINTA2, fontsize=8)
    ax.tick_params(colors=TINTA2, labelsize=7.5, length=0)
    ax.grid(axis="y", color=REGLA, lw=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(REGLA)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG)
    plt.close(fig)


def registrar_fuentes():
    for alias, archivo in (("SS", "SourceSans3-Regular"), ("SSB", "SourceSans3-Bold"), ("SSS", "SourceSans3-Semibold"),
                           ("BIT", "Bitter-SemiBold"), ("BITX", "Bitter-ExtraBold")):
        pdfmetrics.registerFont(TTFont(alias, str(FUENTES / f"{archivo}.ttf")))
    pdfmetrics.registerFontFamily("SS", normal="SS", bold="SSB")


def extraer_logo():
    """Saca el logo de la plantilla del taller (PDF local del docente); sin él, el informe sale sin logo."""
    if LOGO.exists() or not TALLER_PDF.exists():
        return LOGO.exists()
    import pymupdf
    doc = pymupdf.open(str(TALLER_PDF))
    for img in doc[0].get_images(full=True):
        if (img[2], img[3]) == (310, 163):
            pymupdf.Pixmap(doc, img[0]).save(str(LOGO))
            return True
    return False


def recuadro(html):
    """Recuadro con borde superior verde, como el de 'El reto' en la plantilla del taller."""
    est = ParagraphStyle("rq", fontName="SS", fontSize=9.6, leading=12.8, textColor=colors.HexColor(TINTA), alignment=TA_JUSTIFY)
    t = Table([[Paragraph(html, est)]], colWidths=[A4[0] - 2 * MARGEN])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f6f1")), ("LINEABOVE", (0, 0), (-1, 0), 1.6, colors.HexColor(ACENTO)),
                           ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor(REGLA)), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def banda(c, _doc):
    """Banda superior de la primera página: logo, línea dorada, título y datos del documento."""
    w, h = A4
    c.saveState()
    c.setFillColor(colors.HexColor(FONDO))
    c.rect(0, h - BANDA, w, BANDA, stroke=0, fill=1)
    if extraer_logo():
        c.drawImage(str(LOGO), MARGEN, h - 0.55 * cm - 1.3 * cm, width=2.47 * cm, height=1.3 * cm)
    c.setFillColor(colors.HexColor(DORADO))
    c.rect(MARGEN, h - 2.55 * cm, 1.1 * cm, 0.07 * cm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("BITX", 21)
    c.drawString(MARGEN, h - 3.45 * cm, "Defiende el servidor")
    c.setFillColor(colors.HexColor("#b6c1ba"))
    c.setFont("SS", 9.4)
    c.drawString(MARGEN, h - 3.95 * cm, "Informe del taller evaluable: versionar, comparar y recuperar un clasificador de ataques (UNSW-NB15)")
    columnas = [("DOCUMENTO", "Informe técnico"), ("CURSO", "MLOps · 2026-II"), ("FECHA", "30 de octubre de 2026"),
                ("AUTORA", "Lis Alejandra Sánchez Moreno"), ("GRUPO", "B (4:00 p. m. a 6:00 p. m.)")]
    x = MARGEN
    ancho = (w - 2 * MARGEN) / 5.2
    for etiqueta, valor in columnas:
        c.setFillColor(colors.HexColor("#91a59a"))
        c.setFont("SSB", 5.8)
        c.drawString(x, h - 4.55 * cm, " ".join(etiqueta))
        c.setFillColor(colors.white)
        c.setFont("SSB" if etiqueta != "AUTORA" else "SSB", 7.6)
        c.drawString(x, h - 4.9 * cm, valor)
        x += ancho * (1.35 if etiqueta == "FECHA" else 1.0) if etiqueta != "AUTORA" else ancho * 1.25
    c.restoreState()


class LienzoNumerado(lienzo.Canvas):
    """Pie de página de la plantilla: institución a la izquierda, uso académico y 'n / N' a la derecha."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._paginas = []

    def showPage(self):
        self._paginas.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._paginas)
        for estado in self._paginas:
            self.__dict__.update(estado)
            self.pie(total)
            super().showPage()
        super().save()

    def pie(self, total):
        w, _ = A4
        self.setStrokeColor(colors.HexColor(ACENTO))
        self.setLineWidth(0.6)
        self.line(MARGEN, 1.15 * cm, w - MARGEN, 1.15 * cm)
        self.setFillColor(colors.HexColor(TINTA2))
        self.setFont("SSB", 6.4)
        self.drawString(MARGEN, 0.78 * cm, "UNIVERSIDAD EXTERNADO DE COLOMBIA · DEPARTAMENTO DE MATEMÁTICAS")
        self.drawRightString(w - MARGEN, 0.78 * cm, f"USO ACADÉMICO · {self._pageNumber} / {total}")


def construir():
    registrar_fuentes()
    base = ParagraphStyle("b", fontName="SS", fontSize=9.6, leading=12.8, textColor=colors.HexColor(TINTA), alignment=TA_JUSTIFY, spaceAfter=4.5)
    h1 = ParagraphStyle("h1", parent=base, fontName="BIT", fontSize=12, leading=15, textColor=colors.HexColor(VERDE), spaceBefore=9, spaceAfter=4, alignment=0, keepWithNext=1)
    sub = ParagraphStyle("s", parent=base, fontSize=8.4, textColor=colors.HexColor(TINTA2), alignment=0, spaceAfter=2)
    cel = ParagraphStyle("c", parent=base, fontSize=8.3, leading=10.3, alignment=0, spaceAfter=0)
    celb = ParagraphStyle("cb", parent=cel, fontName="SSB", textColor=colors.HexColor(VERDE))
    ref = ParagraphStyle("r", parent=base, alignment=0)

    def tabla(filas, anchos, cabecera=1):
        data = [[Paragraph(str(c), celb if i < cabecera else cel) for c in f] for i, f in enumerate(filas)]
        t = Table(data, colWidths=[a * cm for a in anchos], repeatRows=cabecera)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, cabecera - 1), colors.HexColor(TENUE)),
                               ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.HexColor(ACENTO)),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(REGLA)),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                               ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
        return t

    P = lambda t, s=base: Paragraph(t, s)
    H = []
    H += [P("Código: github.com/Lis-Sanchez-DS/RetoDefenderServidor (público) · Modelos: huggingface.co/LisSanchez-DS/defender-servidor (privado, acceso de lectura para el docente)", sub)]

    # 1. Datos
    H += [P("1. Datos y limpieza", h1),
          P(f"Usé las particiones publicadas de UNSW-NB15 (descarga manual de la página de UNSW el 5 de octubre de 2026). Sus SHA-256 coinciden con los del anexo del taller, y las filas y columnas "
            f"coinciden con lo publicado ({n(tr['filas'])} y {n(te['filas'])} filas, {tr['columnas']} columnas). El objetivo es <i>label</i> (0 normal, 1 ataque), y el ataque es la clase positiva. "
            f"Vemos cómo la mayoría de las filas son ataques (" + pct(tr['clases_label']['1'] / tr['filas']) + " en entrenamiento y " + pct(te['clases_label']['1'] / te['filas']) + " en la prueba) y cómo hay muchísimas copias exactas de las entradas:"),
          tabla([["Archivo", "Filas", "Ataques", "Copias de otra fila (por entradas)", "Grupos con etiquetas contradictorias"],
                 ["training-set", n(tr["filas"]), pct(tr["clases_label"]["1"] / tr["filas"]),
                  f"{n(tr['duplicados_por_entradas']['filas_copia_de_otra'])} ({pct(tr['duplicados_por_entradas']['filas_copia_de_otra'] / tr['filas'])})",
                  f"{tr['duplicados_por_entradas']['grupos_con_etiquetas_contradictorias']} ({n(tr['duplicados_por_entradas']['filas_en_grupos_contradictorios'])} filas)"],
                 ["testing-set", n(te["filas"]), pct(te["clases_label"]["1"] / te["filas"]),
                  f"{n(te['duplicados_por_entradas']['filas_copia_de_otra'])} ({pct(te['duplicados_por_entradas']['filas_copia_de_otra'] / te['filas'])})",
                  f"{te['duplicados_por_entradas']['grupos_con_etiquetas_contradictorias']} ({n(te['duplicados_por_entradas']['filas_en_grupos_contradictorios'])} filas)"]],
                [2.6, 1.8, 1.8, 5.8, 5.2]),
          Spacer(1, 3),
          P(f"<b>Política de limpieza.</b> Las entradas son 42 variables: se excluyen <i>label</i>, <i>attack_cat</i> (revela la respuesta) e <i>id</i> (solo identifica filas, y lo uso para auditar y para el <i>row_id</i> de las predicciones). "
            f"No hay faltantes. Los duplicados no se eliminan: cada vector de entrada idéntico forma un grupo que va entero a entrenamiento o a validación, y los grupos contradictorios se conservan (son error irreducible). "
            f"Se excluyó una fila de entrenamiento con el estado raro «no»; en la prueba no se eliminó ninguna fila, y los estados que el entrenamiento no vio se codifican con ceros. "
            f"El preprocesamiento (estandarización, <i>one-hot</i> de <i>service</i> y <i>state</i>, y <i>proto</i> con una categoría para lo infrecuente) se ajusta solo con entrenamiento y viaja dentro del pipeline guardado."),
          P(f"<b>Partición.</b> Creé entrenamiento y validación solo desde el archivo de entrenamiento, con semilla {aud['particion']['semilla']}: {n(aud['particion']['filas']['entrenamiento'])} y {n(aud['particion']['filas']['validacion'])} filas (70/30), "
            f"con {pct(aud['particion']['ataques_por_conjunto']['entrenamiento'], 2)} de ataques en ambas y {aud['particion']['grupos_repartidos_entre_entrenamiento_y_validacion']} grupos repartidos. Es la misma para todos los experimentos. "
            f"De la prueba (que solo se usó al final) medí el solapamiento: {n(aud['cruces_entre_particiones']['filas_test_con_vector_identico_en_entrenamiento_oficial'])} filas ({n(aud['cruces_entre_particiones']['porcentaje_test'], 1)} %) tienen una entrada idéntica en entrenamiento, "
            f"así que sus cifras serán algo optimistas en esas filas.")]

    # 2. Experimentos
    H += [P("2. Cuatro experimentos, dos familias", h1),
          P("Las hipótesis las escribí antes de entrenar (están en <i>docs/taller_hipotesis.md</i>, con commit previo a cada ejecución). Todas las configuraciones usan la misma partición y las mismas métricas, con el ataque como clase positiva. "
            "Para los denominadores cero, precisión y recall usan <i>zero_division</i> = 0 y la tasa de falsos positivos (FPR) es FP/(FP+TN), que nunca es 0/0 porque siempre hay filas normales. "
            "La línea base (C0) predice siempre la clase dominante, así que alarma todo el tráfico normal. Todas se comparan con umbral 0,5, salvo C4, que trae el suyo."),
          tabla([["ID", "Familia y cambio", "Hipótesis (resumen)", "Umbral", "Recall", "Prec.", "FPR", "F1 macro", "Veredicto"],
                 ["C0", "Línea base (siempre ataque)", "Referencia", "0,5", n(exp.loc['C0'].recall, 3), n(exp.loc['C0'].precision, 3), n(exp.loc['C0'].fpr, 3), n(exp.loc['C0'].macro_f1, 3), "n/a"],
                 ["C1", "Regresión logística, numéricas y categóricas", "Supera con claridad a la línea base y deja de alarmar todo el tráfico normal", "0,5", n(exp.loc['C1'].recall, 3), n(exp.loc['C1'].precision, 3), n(exp.loc['C1'].fpr, 3), n(exp.loc['C1'].macro_f1, 3), "Se cumple"],
                 ["C2", "XGBoost estándar (300 árboles, prof. 6, tasa 0,1)", "Al enfocarse en la precisión, XGBoost supera a la regresión logística", "0,5", n(exp.loc['C2'].recall, 3), n(exp.loc['C2'].precision, 3), n(exp.loc['C2'].fpr, 3), n(exp.loc['C2'].macro_f1, 3), "Se cumple"],
                 ["C3", "XGBoost con Optuna (25 pruebas, solo entrenamiento)", "Mantiene un recall de al menos 0,95 y reduce de forma significativa los falsos positivos", "0,5", n(exp.loc['C3'].recall, 3), n(exp.loc['C3'].precision, 3), n(exp.loc['C3'].fpr, 3), n(exp.loc['C3'].macro_f1, 3), "En parte"],
                 ["C4", "XGBoost con Optuna que busca también el umbral (25 pruebas)", "Con el mejor umbral balancea mejor FP y FN y logra mayor precisión", n(exp.loc['C4'].umbral, 4), n(exp.loc['C4'].recall, 3), n(exp.loc['C4'].precision, 3), n(exp.loc['C4'].fpr, 3), n(exp.loc['C4'].macro_f1, 3), "En parte"]],
                [0.8, 3.5, 4.1, 1.5, 1.3, 1.2, 1.2, 1.5, 1.9]),
          Spacer(1, 3),
          P(f"<b>Interpretación.</b> Pasar de la regresión logística a XGBoost (C1 a C2) es el salto grande: las falsas alarmas bajan de {n(exp.loc['C1'].fp)} a {n(exp.loc['C2'].fp)} sobre {n(exp.loc['C2'].tn + exp.loc['C2'].fp)} filas normales, "
            f"aunque a 0,5 XGBoost pierde más ataques ({n(exp.loc['C2'].fn)} frente a {n(exp.loc['C1'].fn)}). Optuna a 0,5 (C3) casi no cambia nada frente a C2 (precisión {n(exp.loc['C3'].precision, 4)} contra {n(exp.loc['C2'].precision, 4)}, "
            f"{n(exp.loc['C3'].fp)} contra {n(exp.loc['C2'].fp)} falsas alarmas y {n(exp.loc['C3'].fn)} contra {n(exp.loc['C2'].fn)} ataques perdidos), por eso la hipótesis se cumple solo en parte. "
            f"C4 sube la precisión a {n(exp.loc['C4'].precision, 4)} con un umbral de {n(exp.loc['C4'].umbral, 4)}, pero el recall baja a {n(exp.loc['C4'].recall, 4)}: balancea los errores en otra dirección (menos falsas alarmas, más ataques perdidos), y por eso también la califiqué como parcial. "
            f"La búsqueda de C4 podó 16 de 25 pruebas (9 completas), así que exploró poco; con otra semilla el resultado podría variar.")]

    # 3. v1 vs v2
    H += [P("3. Cambios entre v1 y v2", h1),
          P(f"v1 es C2 (XGBoost estándar, umbral {n(v1['umbral'], 2)}) y v2 es C4 (XGBoost de la búsqueda de Optuna, con umbral {n(v2['umbral'], 2)}, que redondeé desde el 0,6607 que halló la búsqueda). Se distinguen en los hiperparámetros y en el umbral, y sus archivos de modelo son distintos. "
            f"Mi hipótesis era que Optuna lograría un mejor desempeño que el XGBoost base, y se cumple en validación, aunque con un matiz que explico después de la tabla:"),
          tabla([["Validación (" + n(sel['v1']['metricas_validacion']['tn'] + sel['v1']['metricas_validacion']['fp'] + sel['v1']['metricas_validacion']['fn'] + sel['v1']['metricas_validacion']['tp']) + " filas)", "Precisión", "Recall", "FPR", "F1 macro", "Falsas alarmas (FP)", "Ataques perdidos (FN)"],
                 [f"v1 (C2, umbral {n(v1['umbral'], 2)})", n(m1['precision'], 4), n(m1['recall'], 4), n(m1['tasa_falsos_positivos'], 4), n(m1['macro_f1'], 4), n(m1['fp']), n(m1['fn'])],
                 [f"v2 (C4, umbral {n(v2['umbral'], 2)})", n(m2['precision'], 4), n(m2['recall'], 4), n(m2['tasa_falsos_positivos'], 4), n(m2['macro_f1'], 4), n(m2['fp']), n(m2['fn'])]],
                [3.9, 1.9, 1.9, 1.9, 1.9, 3.0, 3.1]),
          Spacer(1, 3),
          P(f"<b>Qué mejora y qué empeora.</b> Mejoran la precisión (+{n(m2['precision'] - m1['precision'], 4)}) y las falsas alarmas ({n(m1['fp'])} a {n(m2['fp'])}). Empeoran el recall ({n(m1['recall'], 4)} a {n(m2['recall'], 4)}), el F1 macro y, sobre todo, los ataques perdidos ({n(m1['fn'])} a {n(m2['fn'])}). "
            f"Acepté esa pérdida porque el recall se mantiene por encima del piso de 0,95, aunque reconozco que contradice un poco mi prioridad de no perder ataques.")]
    H += [recuadro(f"<b>Hallazgo importante.</b> La mejora viene del umbral, no de los hiperparámetros. Con umbral 0,5 el modelo de v2 da una precisión de {n(mlf['C4']['metricas']['val_precision'], 4)}, prácticamente igual que la de v1 ({n(m1['precision'], 4)}); y el modelo de v1 con su propio umbral de {n(p95_c2['val_p95_umbral'], 4)} (recall {n(p95_c2['val_p95_recall'], 4)}) "
            f"alcanza precisión {n(p95_c2['val_p95_precision'], 4)} con {n(p95_c2['val_p95_fp'])} falsas alarmas, algo mejor que v2. Es decir, la búsqueda de Optuna no aportó una ventaja medible aquí; la comparación v1/v2 se hizo en validación, como pide el taller.")]

    # 4. Umbral
    H += [P("4. Elección del umbral", h1),
          P(f"El umbral de cada versión lo elegí usando solo validación. Prioricé el recall (un ataque perdido cuesta más que una falsa alarma) con un piso de 0,95, y entre los umbrales que lo cumplen busqué la mayor precisión. "
            f"La figura muestra el modelo de v2 en validación: al subir el umbral la precisión sube y el recall baja, y el recall cruza el piso de 0,95 en {n(mlf['C4']['metricas']['val_p95_umbral'], 3)}, de modo que 0,66 queda un poco antes del cruce, con algo de margen sobre el piso (recall {n(m2['recall'], 4)}). "
            f"v1 se deja en 0,5 (primera solución funcional, sin ajuste). La selección de v1 y v2 se congeló en el commit 4fda613, antes de que existieran las predicciones de la prueba."),
          Image(str(FIG), width=15.2 * cm, height=15.2 * cm * 2.5 / 7.2)]

    # 5. Test
    H += [P("5. Prueba final", h1),
          P(f"Evalué una sola vez v2 (su SHA-256 coincide con el del modelo congelado) sobre las {n(tst['filas_evaluadas'])} filas de la prueba, sin eliminar ninguna, con umbral {n(tst['umbral'], 2)}. "
            f"El archivo <i>predicciones-test.csv</i> tiene una fila por registro (<i>row_id</i>, <i>y_true</i>, <i>score_attack</i>, <i>y_pred</i>, con la regla y_pred = 1 si el puntaje es al menos el umbral) y las métricas se recalculan desde él (hay una prueba automática)."),
          tabla([["", "Recall", "Precisión", "FPR", "F1 macro", "TN", "FP", "FN", "TP"],
                 ["Validación", n(m2['recall'], 4), n(m2['precision'], 4), n(m2['tasa_falsos_positivos'], 4), n(m2['macro_f1'], 4), n(m2['tn']), n(m2['fp']), n(m2['fn']), n(m2['tp'])],
                 ["Prueba", n(mt['recall'], 4), n(mt['precision'], 4), n(mt['tasa_falsos_positivos'], 4), n(mt['macro_f1'], 4)] + [n(tst['matriz_confusion'][k]) for k in ('tn', 'fp', 'fn', 'tp')]],
                [2.4, 1.8, 1.9, 1.8, 1.9, 1.9, 1.9, 1.9, 1.9]),
          Spacer(1, 3),
          P(f"Vemos cómo el recall se sostiene ({n(mt['recall'], 3)}), pero la precisión cae unos {n(100 * (m2['precision'] - mt['precision']), 0)} puntos y la FPR se multiplica por {n(mt['tasa_falsos_positivos'] / m2['tasa_falsos_positivos'], 1)}: casi una de cada cinco filas normales se marca como ataque. "
            f"Por familia, Fuzzers es la más débil (recall {n(tst['recall_por_familia_de_ataque']['Fuzzers'], 3)}) y Analysis le sigue ({n(tst['recall_por_familia_de_ataque']['Analysis'], 3)}); las demás están por encima de {n(np.floor(100 * min(v for k, v in tst['recall_por_familia_de_ataque'].items() if k not in ('Fuzzers', 'Analysis'))) / 100, 2)}. "
            f"No demostré la causa de la caída; una hipótesis (sin comprobar) es que la prueba incluye una sesión de captura ausente del entrenamiento. El resultado se conserva aunque empeore, como pide el taller.")]

    # 6. Limitaciones y trazabilidad
    H += [P("6. Limitaciones, trazabilidad y declaración de ayuda", h1),
          P("<b>Limitaciones.</b> (a) Es un benchmark histórico de laboratorio (2015) y no demuestra eficacia en una red actual. (b) La validación estima rendimiento en distribución, con filas parecidas a ambos lados, así que es optimista. "
            "(c) Este mismo conjunto de prueba ya se había evaluado una vez, con otro modelo, en una versión anterior del trabajo (con una guía previa del taller); no se usó para elegir el modelo ni el umbral actuales, pero conviene decirlo. "
            "(d) Solo comparé dos familias, y la mejora de v2 no depende de los hiperparámetros. (e) Las tarjetas inmutables de v1 y v2 en Hugging Face traen por error los tamaños de una partición 80/20 descartada; la tarjeta principal tiene una fe de erratas y los modelos no cambian."),
          tabla([["", "Código (tag)", "Run MLflow", "Revisión de Hugging Face (SHA)", "SHA-256 del modelo"],
                 ["v1", "v1 (f40c9cc)", v1['run_id'], v1['huggingface']['revision_sha'], v1['modelo_sha256'][:24] + "…"],
                 ["v2", "v2 (f235480)", v2['run_id'], v2['huggingface']['revision_sha'], v2['modelo_sha256'][:24] + "…"]],
                [0.9, 2.3, 4.5, 6.5, 3.4]),
          Spacer(1, 3),
          P(f"<b>Recuperación.</b> En una carpeta y un entorno nuevos (dependencias fijadas, sin cachés), recuperé con DVC los datos originales y preparados (huellas iguales a las del anexo) y descargué v1 y v2 de Hugging Face por su SHA completo. "
            f"Con {rec['filas']} filas fijas de validación, las predicciones de ambas versiones coincidieron con las guardadas y la diferencia máxima de probabilidad fue {n(max(x['diferencia_maxima_en_probabilidad'] for x in rec['versiones'].values()), 1)} (tolerancia declarada: {f"{rec['tolerancia_probabilidades']:.0e}".replace('e-0', 'e-')}), así que la prueba quedó {rec['resultado'].lower()}."),
          P("<b>Uso de IA.</b> Usé Claude Code (Anthropic, modelo Claude Sonnet 5.5) para escribir y refactorizar el código y las pruebas, redactar la documentación y las tarjetas de modelo, ejecutar la prueba de recuperación y redactar este informe a partir de los archivos de la carpeta de evidencias y de mi perfil de estilo de escritura. "
            "Las decisiones (partición, configuraciones, definición de v1 y v2, umbrales), todas las hipótesis y sus veredictos son míos. Revisé el informe y verifiqué sus cifras contra los archivos de evidencia (una prueba automática compara las cifras del PDF con ellos). Los detalles están en <i>entrega.md</i>.")]

    # Referencias (no cuentan para el límite)
    H += [PageBreak(), P("Referencias", h1)]
    for r in ["Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS, IEEE.",
              "Moustafa, N. y Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set. Information Security Journal: A Global Perspective, 25(1-3), 18-31.",
              "Chen, T. y Guestrin, C. (2016). XGBoost: a scalable tree boosting system. KDD.",
              "Akiba, T. et al. (2019). Optuna: a next-generation hyperparameter optimization framework. KDD.",
              "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. JMLR, 12, 2825-2830.",
              "Documentación de Git, DVC, MLflow Tracking y Hugging Face Hub (descarga por revisión y tarjetas de modelo)."]:
        H.append(P(r, ref))
    return H


def main():
    figura()
    w, h = A4
    doc = BaseDocTemplate(str(SALIDA), pagesize=A4, leftMargin=MARGEN, rightMargin=MARGEN, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                          title="Defiende el servidor: informe", author="Lis Alejandra Sánchez Moreno")
    ancho = w - 2 * MARGEN
    doc.addPageTemplates([
        PageTemplate(id="primera", frames=[Frame(MARGEN, 1.6 * cm, ancho, h - BANDA - 0.5 * cm - 1.6 * cm, id="f1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=banda),
        PageTemplate(id="resto", frames=[Frame(MARGEN, 1.6 * cm, ancho, h - 1.5 * cm - 1.6 * cm, id="f2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)])])
    historia = construir()
    historia.insert(0, NextPageTemplate("resto"))
    doc.build(historia, canvasmaker=LienzoNumerado)
    from pypdf import PdfReader
    print("páginas:", len(PdfReader(str(SALIDA)).pages), "(5 de contenido + 1 de referencias como máximo)")


if __name__ == "__main__":
    main()
