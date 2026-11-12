# Práctica 4. Detección de vehículos y matrículas

## Tabla de contenidos
- [Introducción](#introduccion)
- [Pasos previos](#pasos-previos)
- [Parte A](#parte-a-detección-y-seguimiento-de-personas-y-vehículos-y-detección-de-matrículas-yolo-vs-rt-detr)
- [Parte B](#parte-b-lectura-de-matrículas-ocr-easyocr-vs-smolvlm2)
- [Fuentes consultadas](#fuentes-consultadas)

## Introducción

En el cuaderno VC_P4 correspondiente a la [cuarto práctica de la asignatura](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P4) se proponen una serie de tareas que deben ser completadas. A continuación, se presenta una explicación de las distintas tareas propuestas y como se han resuelto para completarlas con éxito.

## Pasos previos
Para la [parte A](#parte-a-detección-y-seguimiento-de-personas-y-vehículos-y-detección-de-matrículas-yolo-vs-rt-detr), creamos un entorno en Anaconda llamado VC_P4:
```bash
conda create --name VC_P4 python=3.9.5
```
Luego, se activa el entorno:
```bash
conda activate VC_P4
```
Una vez activado, se instalan los siguientes paquetes:
```bash
pip install ultralytics lap tensorboard faster_coco_eval
```
Inclyendo esto:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
---
Por problemas de incompatibilidad entre paquetes del entorno anterior para la [parte B](#parte-b-lectura-de-matrículas-ocr-easyocr-vs-smolvlm2), nos vimos forzados a crear un entorno en Anaconda llamado VC_P4b:
```bash
conda create --name VC_P4b python=3.9.5
```
Luego, activamos el entorno:
```bash
conda activate VC_P4b
```
E instalamos las dependencias necesarias:
```bash
pip install pytesseract easyocr paddleocr transformers ultralytics paddlepaddle-gpu num2words
```

## Parte A. Detección y seguimiento de personas y vehículos, y detección de matrículas (YOLO vs. RT-DETR)
 
### Descripción
 
El notebook [P4.ipynb](P4_main.ipynb) procesa un vídeo de tráfico con **YOLOv11n** y posteriormente con **RT-DETR** para:
 
- Detectar y **seguir** personas y vehículos (bicicleta, coche, moto, guagua).
- Detectar **matrículas** dentro del recorte de cada vehículo con un modelo entrenado a medida.
- **Anonimizar** (pixelar) caras y matrículas (solo **YOLOv11n**).
- **Contar** objetos únicos y calcular el **flujo de salidas** por los bordes izquierdo y derecho (solo **YOLOv11n**).
- Generar un **vídeo anotado** y un **CSV** con todas las detecciones.
- **Comparar** los dos detectores de matrículas entrenados con el mismo dataset.


### Detección con YOLOv11

#### Configuración y modelos

Se cargan el detector general (`yolo11n.pt`, COCO) y el detector de matrículas entrenado, y se definen las clases de interés y los parámetros.

```python
source_video = "C0142.mp4"
output_video = "output.mp4"
csv_output = "detections.csv"

# Carga de los modelos: preentrenado y detección de matrículas
general_model = YOLO("yolo11n.pt")
plate_model = YOLO("./runs/detect/yolo_runs/plates_detection/weights/best.pt")

# Definición de las clases de interés
PERSON = 0
VEHICLES = {1, 2, 3, 5}  # bicicleta, coche, moto, guagua
CLASSES = [PERSON, *VEHICLES]
PLATE_VEHICLES = {2, 3, 5}  # vehículos con matrícula

# Parámetros de detección y recorte
conf_coco = 0.4
conf_plate = 0.3
padding = 0.05  # margen extra alrededor del vehículo (5%)

face_ratio = 0.22  # se asume que la fracción superior de la caja es la cara

# Parámetros del flujo
margin = 0.08
min_history = 5
lost_frames = 15
directions = ["izquierda", "derecha"]
```

#### Entrenamiento del detector de matrículas (Anaconda Prompt)

Con el directorio actual en `P4`:

```bash
# GPU (CUDA)
yolo detect train data=data.yaml model=yolo11n.pt imgsz=416 batch=4 device=0 epochs=40 project=yolo_runs name=plates_detection exist_ok=True hsv_h=0.015 hsv_s=0.5 hsv_v=0.3 degrees=5 translate=0.1 scale=0.3 fliplr=0.0 mosaic=0.5

# CPU
yolo detect train data=data.yaml model=yolo11n.pt imgsz=416 batch=4 device=cpu epochs=40 project=yolo_runs name=plates_detection exist_ok=True hsv_h=0.015 hsv_s=0.5 hsv_v=0.3 degrees=5 translate=0.1 scale=0.3 fliplr=0.0 mosaic=0.5
```

#### Funciones auxiliares del CSV

El CSV tiene una fila por objeto y fotograma, con las columnas: `fotograma`, `tipo_objeto`, `confianza`, `identificador_tracking`, `x1..y2`, `matricula_en_su_caso`, `confianza_matricula`, `mx1..my2`, `texto_matricula` (esta última se rellena en `P4b`).

```python
def write_csv_header(path): ...      # crea el CSV con la cabecera
def make_row(frame_n, name, conf, tid, box, plate=None): ...   # construye una fila
def append_rows(path, rows): ...     # añade filas al CSV
```

#### Funciones de dibujado, anonimizado y flujo

**Pixelado** (anonimizado de caras y matrículas): reduce la región y la vuelve a ampliar con interpolación `INTER_NEAREST`.

```python
def pixelate(frame, box, factor=10):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (max(1, rw // factor), max(1, rh // factor)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
```

**Cálculo del flujo**: un objeto cuenta como "salida" si lleva más de `lost_frames` fotogramas sin verse, tiene historial suficiente y su última posición está cerca de un borde lateral.

```python
def exit_direction(cx, cy):
    if cx <= w * margin:
        return "izquierda"
    if cx >= w * (1 - margin):
        return "derecha"
    return None


def update_exits(current_frame):
    for tid in [t for t, f in last_seen.items() if current_frame - f > lost_frames]:
        hist = track_history.pop(tid)
        del last_seen[tid]
        if len(hist) >= min_history and tid not in counted_exit:
            d = exit_direction(*hist[-1])
            if d:
                exit_count[d][track_class[tid]] += 1
                counted_exit.add(tid)
```

Otras funciones: `draw` (caja con etiqueta), `draw_panel` (panel semitransparente) y `render` (paneles de **DETECCIONES** y **FLUJO**).

#### Bucle principal de procesado

Por cada fotograma:

1. `general_model.track(..., persist=True)` devuelve cajas, clases y **IDs de seguimiento**.
2. Se actualizan los conjuntos de IDs únicos y el historial de trayectorias (`deque(maxlen=30)`).
3. Para **personas** se pixela la franja superior de la caja (cara).
4. Para **vehículos con matrícula** se recorta el vehículo con margen, se detecta la matrícula con la mayor confianza y se convierten las coordenadas del recorte a coordenadas del fotograma.
5. Se guardan las filas en el CSV, se actualizan las salidas, se pixelan las cajas y se dibuja el resultado.

```python
if cls in PLATE_VEHICLES:
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * padding)); cy1 = max(0, int(y1 - bh * padding))
    cx2 = min(w, int(x2 + bw * padding)); cy2 = min(h, int(y2 + bh * padding))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size:
        res = plate_model.predict(crop, conf=conf_plate, verbose=True)[0]
        if len(res.boxes):
            i = int(res.boxes.conf.argmax())           # mejor matrícula del recorte
            pconf = float(res.boxes.conf[i])
            px1, py1, px2, py2 = res.boxes.xyxy[i].cpu().numpy()
            plate = (pconf, int(px1 + cx1), int(py1 + cy1),   # coordenadas globales
                            int(px2 + cx1), int(py2 + cy1))
```

Salidas: `output.mp4` y `detections.csv`.

### Detección con RT-DETR

#### Entrenamiento

Mismo dataset (`data.yaml`) para poder comparar. Se puede entrenar por línea de comandos (`rtdetr-l.pt`, `imgsz=416`, `epochs=40`, mismos hiperparámetros de aumento) o desde el notebook; esta celda se omite si ya existe `best.pt`:

```python
data_yaml = "./data.yaml"
rtdetr_plate_path = "./runs/detect/rtdetr_runs/plates_detection/weights/best.pt"

if not os.path.exists(rtdetr_plate_path):
    rtdetr = RTDETR("rtdetr-l.pt")
    rtdetr.train(data=data_yaml, epochs=1, imgsz=640, batch=8, seed=0,
                 project="rtdetr_runs", name="plates_detection")
else:
    print("Ya existe un modelo RT-DETR entrenado: ", rtdetr_plate_path)
```

Validación rápida:

```python
rtdetr_metrics = RTDETR(rtdetr_plate_path).val(data=data_yaml, imgsz=640, verbose=False)
# precision, recall, mAP50, mAP50-95
```

#### Procesado del vídeo

Se reinician el tracker y los contadores y se vuelve a procesar el vídeo. **El detector general sigue siendo YOLOv11n**: así las detecciones de vehículos son idénticas en ambas ejecuciones y solo cambia el detector de matrículas, lo que hace la comparación justa.

```python
general_model = YOLO("yolo11n.pt")                      # tracker nuevo
rtdetr_plate_model = RTDETR(rtdetr_plate_path)          # único cambio: modelo de matrículas
...
res = rtdetr_plate_model.predict(crop, conf=conf_plate, verbose=False)[0]
```

Este bucle es una versión simplificada (sin pixelado ni panel de flujo): mantiene detección y seguimiento, detección de matrículas, conteo, vídeo y CSV.

Salidas: `output_rtdetr.mp4` y `detections_rtdetr.csv`.

### Comparativa YOLO vs. RT-DETR

#### Métricas de validación

Se valida cada modelo con el mismo `data.yaml` e `imgsz=640` y se calcula precisión, recall, **F1**, **mAP50**, **mAP50-95** y tiempo de inferencia.

```python
def validate(model, name):
    v = model.val(data=data_yaml, imgsz=640, plots=False, verbose=False)
    p, r = float(v.box.mp), float(v.box.mr)
    return {
        "modelo": name, "precision": p, "recall": r,
        "F1": 2 * p * r / (p + r) if (p + r) else 0.0,
        "mAP50": float(v.box.map50), "mAP50-95": float(v.box.map),
        "inferencia_ms/img": v.speed["inference"],
    }
```

#### Curvas de entrenamiento

Gráficas de mAP50 y mAP50-95 por época a partir del `results.csv` de cada entrenamiento (las curvas PR/F1 y la matriz de confusión ya las guarda Ultralytics en cada carpeta de entrenamiento).

#### Métricas sobre el vídeo (sin anotaciones)

Calculadas desde los CSV; son **indicadores de comportamiento, no de precisión real**:

| Métrica | Significado |
|---|---|
| `tasa_deteccion_matricula` | Proporción de detecciones de vehículo con matrícula |
| `conf_media_matricula` | Confianza media de las matrículas detectadas |
| `pct_vehiculos_con_matricula` | % de vehículos únicos con alguna matrícula |
| `estabilidad_media` | Fracción de fotogramas con matrícula por vehículo |
| `parpadeos_medios_por_vehiculo` | Veces que la matrícula aparece y desaparece (Si → No) |

#### Acuerdo entre modelos

Como las detecciones de vehículos coinciden, se cruzan por `fotograma` + `identificador_tracking`. Se muestra una **tabla de acuerdo** YOLO vs. RT-DETR (Si/No) y el **IoU medio** entre cajas de matrícula cuando ambos detectan.

#### Resumen

```python
summary = pd.concat([val_df.T, video_df], axis=0)
summary.round(4)
```

### Archivos generados

| Archivo | Descripción |
|---|---|
| `output.mp4` / `detections.csv` | Vídeo y CSV con YOLO |
| `output_rtdetr.mp4` / `detections_rtdetr.csv` | Vídeo y CSV con RT-DETR |
| `runs/detect/yolo_runs/...` / `runs/detect/rtdetr_runs/...` | Entrenamientos y pesos `best.pt` |

---

## Parte B. Lectura de matrículas (OCR): EasyOCR vs. SmolVLM2

### Descripción

Continuación de la práctica 4. Partiendo de las matrículas detectadas con **RT-DETR**, se lee su texto con dos alternativas y se comparan en **tiempo de inferencia y acierto**:

| Modelo | Tipo | Descripción |
|---|---|---|
| **EasyOCR** | OCR clásico | Redes de detección y reconocimiento de texto (`es`, `en`) |
| **SmolVLM2-500M-Instruct** | VLM (visión-lenguaje) | Modelo multimodal al que se le pide que lea la matrícula |

### Carga de modelos

Se cargan el detector de matrículas (RT-DETR entrenado en la práctica 4), EasyOCR y SmolVLM2. El dispositivo (`cuda`/`cpu`) se elige automáticamente.

```python
val_images_dir = Path("./OCR_Dataset/val/images")
plate_model = RTDETR("./runs/detect/rtdetr_runs/plates_detection/weights/best.pt")

# OCR: EasyOCR
easy_reader = easyocr.Reader(['es', 'en'], gpu=torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# VLM: SmolVLM2
model_id = "HuggingFaceTB/SmolVLM2-500M-Instruct"
proccessor = AutoProcessor.from_pretrained(model_id)
vlm_model = (
    AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        _attn_implementation="sdpa",
    )
    .to(device)
    .eval()
)
```

### Funciones de inferencia

Para que la comparación sea justa, **ambos modelos reciben el mismo preprocesado** (ampliación ×2 con interpolación cúbica) y **la misma normalización** de la salida (mayúsculas, solo `A-Z0-9`).

### Preprocesado, normalización y distancia de Levenshtein

```python
SCALE = 2
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def prep(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)

def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def lev(a, b):   # distancia de edición, base del CER
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
```

### EasyOCR

Restringe los caracteres permitidos con `allowlist` y mide solo el tiempo de la inferencia.

```python
def infer_easyocr(img):
    img = prep(img)
    start = time.perf_counter()
    result = easy_reader.readtext(img, detail=0, allowlist=ALLOW)
    text = norm("".join(result))
    return text, (time.perf_counter() - start) * 1000
```

### SmolVLM2

El prompt pide específicamente leer la matrícula y responder solo con letras y dígitos; la generación se limita a 15 tokens.

```python
def infer_smolvlm2(img):
    image = Image.fromarray(cv2.cvtColor(prep(img), cv2.COLOR_BGR2RGB))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text",
             "text": "Read the license plate. Reply ONLY with its letters and digits, without any extra words."},
        ],
    }]
    start = time.perf_counter()
    prompt = proccessor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proccessor(text=prompt, images=[image], return_tensors="pt").to(device)

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=15)
    generated_texts = proccessor.batch_decode(generated_ids, skip_special_tokens=True)

    text = norm(generated_texts[0].split("Assistant:")[-1])
    return text, (time.perf_counter() - start) * 1000
```

#### Recortes reales y etiquetado (*ground truth*)

El dataset solo trae las cajas, no el texto, así que se etiqueta a mano.

- `gt_crop` recorta la matrícula con la **anotación real** del dataset (formato YOLO normalizado), no con la del detector. Así se evalúan **solo los OCR**, sin influencia de los fallos de detección, y ambos modelos ven exactamente los mismos recortes.
- Para cada imagen se muestra el recorte y se escribe la matrícula: Enter vacío = ilegible (se descarta) y `q` = salir.
- Cada respuesta se guarda al instante en `ground_truth.csv` (`imagen,texto`). Al volver a ejecutar la celda se saltan las ya etiquetadas.

```python
def gt_crop(img_path):
    frame = cv2.imread(str(img_path))
    lf = labels_dir / (img_path.stem + ".txt")
    if frame is None or not lf.exists():
        return None
    lines = lf.read_text().strip().splitlines()
    if not lines:
        return None
    _, cx, cy, bw, bh = map(float, lines[0].split()[:5])
    H, W = frame.shape[:2]
    x1, y1 = max(0, int((cx - bw / 2) * W)), max(0, int((cy - bh / 2) * H))
    x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None
```

### Ejecución

1. Se cargan las etiquetas y se generan los recortes de las imágenes etiquetadas.
2. **Calentamiento** (3 pasadas sin medir) de cada modelo, para que la primera inferencia no distorsione los tiempos.
3. Cada recorte pasa por ambos modelos en las mismas condiciones.

```python
# Calentamiento (no se mide)
first = next(iter(crops.values()))
for _ in range(3):
    infer_easyocr(first)
    infer_smolvlm2(first)

results = []
for name, crop in crops.items():
    text_easy, time_easy = infer_easyocr(crop)
    text_smol, time_smol = infer_smolvlm2(crop)
    results.append({
        "Imagen": name, "Texto_Real": gt[name],
        "Texto_EasyOCR": text_easy, "Tiempo_EasyOCR_ms": time_easy,
        "Texto_SmolVLM2": text_smol, "Tiempo_SmolVLM2_ms": time_smol,
    })
```

**Condiciones del experimento:** 81 imágenes del conjunto de validación, mismos recortes, mismo preprocesado y normalización, calentamiento previo, GPU NVIDIA GeForce RTX 5070 Laptop (8 GB), Python 3.9.21, PyTorch 2.8.0 + CUDA 12.8.

---

### Métricas

| Métrica | Descripción |
|---|---|
| **Exact match (%)** | Matrículas leídas exactamente igual que la real |
| **CER (%)** | Errores por carácter (inserciones, borrados y sustituciones) sobre el total de caracteres reales. Es menos estricta que el exact match; menor es mejor |
| **Tiempo medio / mediana / desviación (ms)** | Tiempo de inferencia por imagen. La mediana es más representativa si hay valores atípicos |

```python
rows.append({
    "Modelo": m,
    "Exact match (%)": (pred == real).mean() * 100,
    "CER (%)": sum(lev(p, r) for p, r in zip(pred, real)) / real.str.len().sum() * 100,
    "Tiempo medio (ms)": t.mean(),
    "Tiempo mediana (ms)": t.median(),
    "Tiempo std (ms)": t.std(),
})
```

Los resultados por imagen se guardan en `comparativa_ocr.csv`. La tabla resumen se muestra con `display(comparativa.round(2))`.

### Vídeo con OCR

Se recorre el vídeo de detecciones RT-DETR fotograma a fotograma. Para cada detección con matrícula (`matricula_en_su_caso == "Si"`) se recorta la región con `mx1, my1, mx2, my2`, se lee con **EasyOCR**, se guarda el texto en la columna `texto_matricula` y se dibuja sobre el fotograma.

```python
for index, row in detecciones_frame.iterrows():
    if row["matricula_en_su_caso"] == "Si":
        mx1 = max(0, int(row["mx1"])); my1 = max(0, int(row["my1"]))
        mx2 = min(w, int(row["mx2"])); my2 = min(h, int(row["my2"]))

        crop_matricula = frame[my1:my2, mx1:mx2]

        if crop_matricula.size > 0:
            texto_final, _ = infer_easyocr(crop_matricula)
            df.at[index, "texto_matricula"] = texto_final
            cv2.putText(frame, texto_final, (mx1, my1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
```

| Archivo | Tipo | Descripción |
|---|---|---|
| `output_rtdetr.mp4` | Entrada | Vídeo de detecciones RT-DETR (práctica 4) |
| `detections_rtdetr.csv` | Entrada | Detecciones por fotograma |
| `output_rtdetr_ocr.avi` | Salida | Vídeo con el texto leído superpuesto (XVID) |
| `detections_rtdetr_ocr.csv` | Salida | CSV con la columna `texto_matricula` rellena |

## Fuentes consultadas

- [Documentación de OpenCV](https://docs.opencv.org/4.13.0/d6/d00/tutorial_py_root.html)
- [Documentación de numpy](https://numpy.org/doc/stable/reference/module_structure.html)
- [Vídeo sobre labelme](https://www.youtube.com/watch?v=HiW3qeJVQCg)