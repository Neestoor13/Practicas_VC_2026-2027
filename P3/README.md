# Práctica 3. Funciones básicas de OpenCV

## Tabla de contenidos
- [Introducción](#introduccion)
- [Pasos previos](#pasos-previos)
- [Tarea 1](#tarea-1)
    - [Imagen Ideal](#imagen-ideal)
    - [Detección por Webcam](#detección-por-webcam)
- [Tarea 2](#tarea-2)
- [Fuentes consultadas](#fuentes-consultadas)

## Introducción

En el cuaderno VC_P3 correspondiente a la [tercera práctica de la asignatura](https://github.com/otsedom/otsedom.github.io/tree/main/VC/P3) se proponen una serie de tareas que deben ser completadas. A continuación, se presenta una explicación de las distintas tareas propuestas y como se han resuelto para completarlas con éxito.

## Pasos previos

Previo a la realización de las tareas, se deberá realizar cambios al entorno configurado en la práctica 1. Este cambio reside en realizar la instalación de los paquetes _scikit-learn_ y _seaborn_ tal y como se realiza a continuación.

```
pip install scikit-learn seaborn
```

Una vez completada la instalación se podrá comenzar sin inconveniente alguno la práctica.

## Tarea 1

Para la correcta compleción de esta tarea, se tratarán dos situaciones claves:

- Imagen ideal.
- Imagen captada con webcam.

### Imagen ideal

En esta primera situación se partirá de la imagen [**"Monedas.jpg"**](Monedas.jpg), proporcionada en el guión de la práctica. 

<div align="center">

![Imagen ideal de las monedas](Monedas.jpg)

</div>

Con la imagen cargada (y su transformación a RGB), se establecerá en primer lugar las relaciones que se deberán tener en cuenta para la correcta detección. En este caso, se establecen 2 diccionarios (véase la declaración de los mismos más abajo):

- MONEDAS: establece la relación entre el tipo de moneda (2 euros, 1 euro...) y el diámetro de la misma.
- VALORES: establece la relación entre el tipo de moneda y el valor decimal de las diferentes monedas.

```
MONEDAS = {
    "2 euros": 25.75,
    "1 euro": 23.25,
    "50 cent": 24.25,
    "20 cent": 22.25,
    ...
}

VALORES = {
    "2 euros": 2.00,
    "1 euro": 1.00,
    "50 cent": 0.50,
    "20 cent": 0.20,
    ...
}
```

Con la imagen una vez cargada, se transformará esta a escala de grises y se hará uso de la función _medianBlur_ para reducir el ruido que pueda llegar a tener la imagen. Acto seguido, se procederá a la búsqueda de posibles círculos a partir de los gradientes haciendo uso de la función [_HoughCircles_](https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d), con la cual se controlará configurando sus parámetro, valores como el radio mínimo, máximo... 

A continuación se procede a aplicar umbralización de Otsu para obtener una imagen binaria y posteriormente, se comienza con la búsqueda de contornos externos: si el centro de alguno de los círculo detectados cae dentro de uno de estos contornos, se ajustará una elipse para refinar el centro y radio del mismo.

Por último, a partir de una moneda de referencia (en este caso, de 2 euros) se calcula la relación existente entre el diámetro y los píxeles, de tal forma que se pueda calcular el diámetro del resto de detecciones. Asimismo, se comprobará si la diferencia respecto a los diámetro establecidos en _MONEDAS_ supera el valor de _TOLERANCIA_ establecido (0.6): si la supera, se la catalogará como "?"; en caso contrario, se suma al valor total en la escena (por defecto, 0.0). Finalizada esta comprobación, se procederá al dibujado de los círculos, nombres, diámetros estimados y valores de las monedas detectadas en la escena.

<div align="center">

![Monedas con sus valores y diámetros](Calculo_valor.jpg)

</div>

### Detección por webcam

A partir de esta versión ideal, se procederá a realizar la tarea para la detección de las monedas a partir de una webcam. 

NOTA: Se deberá tener en cuenta que los resultados dependerán de valores como la iluminación, la distancia de las monedas a la cámara, la propia cámara...

Para el planteamiento de esta situación, se han definido distintas funciones auxiliares que gestionan distintas responsabilidades en la detección. A continuación, se procederá a realizar un análisis del papel que cumplen estas funciones en la detección de las monedas.

- Selección de referencia interactiva

    La función on_mouse() se encarga de guardar las coordenadas del clic izquierdo del ratón. Con esta función, se podrá seleccionar una moneda de referencia previamente establecida (por ejemplo, 1 euro). De este modo, el código seleccionará la detección más cercana al punto seleccionado para posteriormente realizar el cálculo de la relación entre milímetros y píxeles mencionada anteriormente.

    ```
    def on_mouse(evento, x, y, flags, param):
        global clic
        if evento == cv2.EVENT_LBUTTONDOWN:
            clic = (x, y)
    ```


- Segmentación del folio

    El entorno de trabajo en el que se han realizado las pruebas de detección consistía de un fondo color madera, por lo que muchas de las pruebas realizadas no llegaban a detectar las monedas debido al poco contraste existente. Es por ello que para paliar este problema, se ha planteado esta función.

    La función segmentar_folio proporciona una región delimitada de trabajo y el contraste suficiente para poder aplicar el umbral de Otsu. Para ello, previamente se ha debido de convertir el fotograma capturado a gris y aplicar _medianBlur_, de manera análoga al planteamiento que se siguió para la versión estática. Calculado el umbral, se procederá a la búsqueda de contornos externos y la selección de aquella área de mayor tamaño (el folio).

    ```
    def segmentar_folio(img):
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pimg = cv2.medianBlur(gris, 7)
        _, bina = cv2.threshold(pimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # El folio es la mayor región blanca: se rellena para delimitar la zona de trabajo
        cont_f, _ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cont_f:
            return None
        folio = np.zeros_like(bina)
        cv2.drawContours(folio, [max(cont_f, key=cv2.contourArea)], -1, 255, -1)

        # Monedas = dentro del folio y no blancas
        return cv2.bitwise_and(folio, cv2.bitwise_not(bina))
    ```
    Si no se encuentran contornos, la función devolverá None. Con esta estrategia se presupone que el folio es la región blanca de mayor tamaño capturada por la webcam y que las monedas contrastan con él. Se deberá tener en cuenta que un fondo es irregular, el folio parcialmente fuera de la visión de la webcam o la presencia de muchas sombras pueden suponer que no se produzca la detección correctamente.


- Detección

    La función detectar() combina la segmentación comentada anteriormente junto con morfología, geometría y anaálsi cromático.

    Esta función se encarga de convertir el fotograma captado por la webcam a HSV para realizar un análisis del color captado. Asimismo, obtiene la máscara de las monedas haciendo uso de la función segmentar_folio comentada anteriormente, y aplica apertura morfológica y cierrepara la eliminación de pequeños residuos y cierre para cubrir posibles huecos (por ejemplo, el centro de las monedas de 1/2 euros).

    Extrae además los contornos externos, descartando aquellos muy pequeños o que no cumplan los criterios válidos para ser una moneda. Acto seguido aplica filtros para descartar objetos alargados o poco circulares para, finalmente, realizar el análisis de color y conservar aquellas detecciones que no hayan sido rechazadas.

    ```
    def detectar(img):
        # Imagen en HSV para la discriminación por color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        bina = segmentar_folio(img)
        if bina is None:
            return []

        # Limpieza común
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bina = cv2.morphologyEx(bina, cv2.MORPH_OPEN, k)
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # cierra el centro plateado del 1 € y 2 €
        bina = cv2.morphologyEx(bina, cv2.MORPH_CLOSE, k2)

        # Localiza contornos externos
        contornos, _ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lado = min(img.shape[:2])
        r_min, r_max = 0.015 * lado, 0.15 * lado  # radio como fracción del frame

        det_monedas = []
        for c in contornos:
            if len(c) < 5:
                continue
            area = cv2.contourArea(c)
            per = cv2.arcLength(c, True)
            if per == 0 or area == 0:
                continue
            circularidad = 4 * np.pi * area / per**2

            (x_coor, y_coor), (eje_a, eje_b), _ = cv2.fitEllipse(c)
            if min(eje_a, eje_b) == 0:
                continue
            det_radio = (eje_a + eje_b) / 4

            if (r_min <= det_radio <= r_max and circularidad > 0.8
                    and min(eje_a, eje_b) / max(eje_a, eje_b) > 0.85):
                fam = familia_color(hsv, x_coor, y_coor, det_radio)
                if fam is not None:
                    det_monedas.append((x_coor, y_coor, det_radio, fam))
        return det_monedas
    ```


- Análisis cromático

    Tal y como se mencionó anteriormente, en este código se hace uso de una función de análisis de color para distinguir entre los distintos tipos de moneda.

    Es importante destacar, que se habrán de definir tanto los umbrales de color a tener en cuenta como las familias a las que pertenecen cada una de estas monedas. 

    ```
    # Definición de los umbrales de color
    S_MIN = 40 
    H_COBRE_MAX = 20 
    H_FRIO_MIN, H_FRIO_MAX = 75, 150
    UMBRAL_BI_H = 8
    UMBRAL_BI_S = 50

    # Relación monedas-color

    FAMILIA = {
        "1 cent": "cobre", 
        "2 cent": "cobre", 
        "5 cent": "cobre",
        "10 cent": "oro", 
        "20 cent": "oro", 
        "50 cent": "oro",
        "1 euro": "bi", 
        "2 euros": "bi",
    }
    ```

    Una vez ha quedado esto establecido, se procederá a hablar de la función familia_color().

    Tal y como ha quedado reflejado en la variable global FAMILIA, esta asigna a cada moneda la familia a la que pertenece para, a través del estudio de una región del centro de la moneda detectado en HSV, clasificar a que familia pertenece cada moneda detectada.

    De esta forma, se excluirán píxeles demasiado oscuros o brillantes (correspondientes a sombras o reflejos) y se realizará un análisis comparativo de las medianas de tono y saturación en el centro y en regiones cercanas al borde de la moneda. Asimismo, se descartarán colores fríos muy saturados (verdes, azules...) y, se considerará que un cambio apreciable entre el centro y la corona de la moneda catalogará a la misma como una moneda _bimetálica_ (1/2 euros). En caso contrario, el tono de la corona se utilizará para determinar si la moneda detectada es de cobre u oro.

    Finalmente si no se ha conseguido catalogar el color, ya sea por ausencia de suficientes píxeles o el color es poco fiable, se retornará "?". Para el caso en que el color sea ajeno al de una moneda (entre ellos, los mencionados tonos fríos) se retornará "None".

    ```
    def familia_color(hsv, x_coor, y_coor, det_radio):
        """Devuelve 'cobre', 'oro', 'bi', '?' (color no fiable) o None si es color frío."""
        x, y, r = int(x_coor), int(y_coor), int(det_radio)
        x0, y0 = max(x - r, 0), max(y - r, 0)
        roi = hsv[y0:y + r, x0:x + r]
        if roi.size == 0:
            return None

        yy, xx = np.ogrid[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]]
        dist = np.hypot(xx - x_coor, yy - y_coor)
        util = (roi[..., 2] > 60) & (roi[..., 2] < 240)   # sin sombras ni brillos especulares
        centro = (dist <= 0.4 * det_radio) & util
        # Corona pegada al borde: evita la frontera plata/oro del 1 € y 2 €
        corona = (dist >= 0.82 * det_radio) & (dist <= 0.94 * det_radio) & util
        if centro.sum() < 10 or corona.sum() < 10:
            return "?"

        h_c, s_c, _ = np.median(roi[centro], axis=0)
        h_k, s_k, _ = np.median(roi[corona], axis=0)

        # Color poco saturado: no es fiable, que decida el tamaño
        if s_c < S_MIN and s_k < S_MIN:
            return "?"

        # Descarta colores fríos (azules, verdes...) cuando están saturados
        for h, s in ((h_c, s_c), (h_k, s_k)):
            if s >= S_MIN and H_FRIO_MIN <= h <= H_FRIO_MAX:
                return None

        # Bimetálica: cambio de tono con color suficiente en ambas zonas, o gran cambio de saturación
        dh = min(abs(h_c - h_k), 180 - abs(h_c - h_k))
        if (dh > UMBRAL_BI_H and min(s_c, s_k) > S_MIN) or abs(s_c - s_k) > UMBRAL_BI_S:
            return "bi"

        return "cobre" if h_k < H_COBRE_MAX else "oro"
    ```


- Seguimiento temporal

    Una vez se ejecute el bucle de código encargado de la detección de las monedas, la webcam estará capturando continuamente las mismas monedas en múltiples fotogramas: esto implicada que en cada fotograma clasificado se pueden producir parpadeos en las etiquetas asociadas a las monedas. Con el fin de reducir dichos parpadeos, se ha propuesto el uso de las funciones auxiliares actualizar_tracks() y estado_track().

    Para el correcto funcionamiento de estas funciones, se han declarado previamente unas variables indispensables.

    ```
    N_HIST = 15
    MIN_HIST = 5
    MAX_PERDIDOS = 10
    tracks = []
    ```

    En lo referente a estas variables definidas:

    - `N_HIST`: establece los frames que se recuerdan por moneda detectada.
    - `MIN_HIST`: establece el número mínimo de observaciones para poder mostrar una clasificación.
    - `MAX_PERDIDOS`: establece el número de fotogramas mínimo sin ver una moneda antes de olvidarla.
    - `tracks`: se utiliza para almacenar el seguimiento de las monedas detectadas.

    Para asociar cada detección de las monedas en el _frame_ con el track cuyo centro está suficientemente cerca se hará uso de la función actualizar_tracks(); si no existiera track, se creará uno nuevo. Además, es el encargado de actualizar los valores de los _tracks_ y eliminar aquellos que lleven demasiado tiempo ausentes.
    En cambio, la función estado_track() es quién se encarga de devolver la mediana de los radios y la familia más votada.

    Algo importante a clarificar es que si las monedas se cruzaran, se solaparan o se produjeran movimientos bruscos en la escena, dos _tracks_ podrían llegar a confundirse.

Una vez clarificado el uso que tienen las anteriores funciones en el bucle principal de detección, se procederá a comentar el funcionamiento de este.

Para hacer posible la detección, se hace uso de la función VideoCapture() que incopora OpenCV para capturar la escena de las monedas. Capturada, se invertirá el fotograma detectado y se procesará haciendo uso de la función _detectar()_.

Acto seguido, se calibrará la relación entre mílimetros y píxeles al hacer click sobre la moneda de referencia. Realizada la calibración se procederá a la actualización de los tracks (actualizar_tracks()), ignorando aquellos que aún no cumplen los requisitos mínimos.

A continuación, se realizará el cálculo del diámetro para las monedas detectadas según la relación establecida por la moneda de referencia. Tras esto, se procederá a comprobar que se cumple el requisito de que el diámetro nominal esté por debajo del valor de tolerancia establecido.

No obstante, puede que solo la denominación con el diámetro no sea suficiente, por lo que se aplicará el análisis cromático de forma complementaria (únicamente se ejecuta si existen varias posibilidades para una misma moneda) para intentar determinar a que familia cromática pertenece (cobre, oro, bimetálica).

Por último, se seleccionará la denominación con menor diferencia de diámetro y se dibujará el círculo y la identificación de la moneda sobre la misma, así como se calculará el valor total observado en la escena. En el caso de que no haya candidatas compatibles, se marcará la detección como "?" y no se sumará valor alguno al total detectado.

```
REF = "1 euro" # Se establece la moneda de referencia 
TOLERANCIA = 0.8   # Se aumenta la tolerancia (mm) respecto a la situación ideal

px_por_mm = None
clic = None

vid = cv2.VideoCapture(2)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Deteccion de monedas")
cv2.setMouseCallback("Deteccion de monedas", on_mouse)

while True:
    ret, frame = vid.read()

    if ret:
        img = cv2.flip(frame, 1)

        det_monedas = detectar(img)

        # Calibración: píxeles por milímetro a partir de la moneda de referencia clicada
        if clic and det_monedas:
            x_ref, y_ref, r_ref, _ = min(
                det_monedas, key=lambda m: np.hypot(m[0] - clic[0], m[1] - clic[1])
            )
            if np.hypot(x_ref - clic[0], y_ref - clic[1]) <= r_ref:
                diam_ref_px = 2 * r_ref
                px_por_mm = diam_ref_px / MONEDAS[REF]
        clic = None

        # Seguimiento temporal: acumula historial de radio y familia por moneda
        actualizar_tracks(det_monedas)

        if px_por_mm:
            # Inicialización del valor total de monedas detectadas
            total = 0.0

            for t in tracks:
                if not t["visto"] or len(t["radios"]) < MIN_HIST:
                    continue
                r_med, fam = estado_track(t)
                x_coor, y_coor, det_radio = t["x"], t["y"], r_med
                diam_mm = 2 * det_radio / px_por_mm

                # 1º el tamaño: candidatas compatibles por diámetro
                candidatas = {n: d for n, d in MONEDAS.items() if abs(d - diam_mm) <= TOLERANCIA}

                # 2º el color solo desempata si hay más de una candidata
                if len(candidatas) > 1 and fam != "?":
                    por_color = {n: d for n, d in candidatas.items() if FAMILIA[n] == fam}
                    if por_color:
                        candidatas = por_color

                if candidatas:
                    nombre, diam_real = min(candidatas.items(), key=lambda m: abs(m[1] - diam_mm))
                    total += VALORES[nombre]
                    color = (0, 255, 0)
                else:
                    nombre = "?"
                    color = (0, 0, 255)

                cv2.circle(img, (int(x_coor), int(y_coor)), int(det_radio), color, 2)

                # Texto sobre la moneda detectada
                cv2.putText(
                    img,
                    f"{nombre} ({diam_mm:.1f} mm)",
                    (int(x_coor - det_radio), int(y_coor)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            cv2.putText(img, f"Valor total: {total:.2f} EUR", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            for x_coor, y_coor, det_radio, fam in det_monedas:
                cv2.circle(img, (int(x_coor), int(y_coor)), int(det_radio), (0, 255, 255), 2)
            cv2.putText(img, f"Clic sobre una moneda de {REF}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(img, "r recalibrar | ESC salir",
                    (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Deteccion de monedas", img)

    tecla = cv2.waitKey(20) & 0xFF
    if tecla == 27:  # ESC
        break
    elif tecla == ord("r"):  # Recalibra con la moneda de referencia
        px_por_mm = None
        tracks = []

vid.release()
cv2.destroyAllWindows()
```
## Tarea 2

## Fuentes consultadas

- [Documentación de OpenCV](https://docs.opencv.org/4.13.0/d6/d00/tutorial_py_root.html)
- [Documentación de numpy](https://numpy.org/doc/stable/reference/module_structure.html)
