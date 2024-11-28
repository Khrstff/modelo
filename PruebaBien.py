import cv2
import numpy as np
from rknn.api import RKNN

# Ruta al modelo RKNN
MODEL_PATH = 'path_to_your_model.rknn'

# Inicializar la red RKNN
rknn = RKNN()

# Cargar el modelo RKNN
ret = rknn.load_rknn(MODEL_PATH)
if ret != 0:
    print("Error al cargar el modelo RKNN.")
    exit(ret)

# Inicializar el entorno de la NPU
ret = rknn.init_runtime()
if ret != 0:
    print("Error al inicializar el runtime de la NPU.")
    exit(ret)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Umbral de confianza para detección
CONFIDENCE_THRESHOLD = 0.3

# Mapa de clases de ejemplo (deberías adaptarlo según las clases de tu modelo)
class_map = {0: 'background', 1: 'person', 2: 'car', 3: 'truck', 4: 'motorbike'}

# Función para dibujar los cuadros de detección
def draw_boxes(frame, boxes, confidences, class_ids):
    for i in range(len(boxes)):
        if confidences[i] > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = boxes[i]
            label = class_map[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)  # Color verde para los cuadros
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Función para realizar la inferencia en la NPU
def infer_on_npu(frame):
    # Redimensionar la imagen para que sea compatible con la entrada del modelo
    h, w, _ = frame.shape
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)  # Añadir dimensión de batch

    # Ejecutar la inferencia en la NPU
    outputs = rknn.infer(inputs=[input_data])

    # Extraer los resultados de la inferencia
    detections = outputs[0]  # Aquí, deberías ajustar la extracción según el formato del modelo

    boxes = []
    confidences = []
    class_ids = []

    # Procesar las detecciones
    for detection in detections[0, 0, :, :]:
        confidence = detection[2]
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detection[1])
            box = detection[3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            boxes.append((x1, y1, x2, y2))
            confidences.append(confidence)
            class_ids.append(class_id)

    return boxes, confidences, class_ids

# Bucle de captura y detección
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la inferencia en la NPU
    boxes, confidences, class_ids = infer_on_npu(frame)

    # Dibujar las detecciones en la imagen
    draw_boxes(frame, boxes, confidences, class_ids)

    # Mostrar la imagen con las detecciones
    cv2.imshow("Detección de Objetos (NPU)", frame)

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Finalizar el uso de la NPU
rknn.release()
