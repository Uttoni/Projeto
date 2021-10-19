import cv2
import numpy as np

# Vetor para receber todas as 80 classes do coco.names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define uma cor aleatoria para cada classe
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Carrega a Imagem
img = cv2.imread("Images/animais.jpg")

# Configura a escala da imagem, caso desejado mude o fx e fy
img = cv2.resize(img, None, fx=1, fy=1)

# Pega altura, largura e canais da imagem
height, width, channels = img.shape

# Realiza o Blob da imagem (realiza subtração e escalonamento medio de imagem)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Carrega os arquivos do Yolo e cria a rede
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
# Configura a rede neural de acordo com o YoloVx utilizado na geração da var. net
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Passa o Blob para configuração da rede
net.setInput(blob)
# outs recebe o retorno de todas as detecções
outs = net.forward(output_layers)

# Vertores auxiliares para mostrar resultados na tela
class_ids = []
confidences = []
boxes = []

# Confiança desejada
confidence_expect = 0.5

# Percorre tudo que encontrou na imagem
for out in outs:
    for detection in out:
        scores = detection[5:]
        # Pega id da classe com maior pontuação de confiança
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_expect:
            # Centro x e y do objeto encontrado
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            # Largura e altura do objeto
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordenadas inicial do objeto (superior esquerdo)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Guarda informações necessarias
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# NMSBoxes utilizado para minimizar resultados sobre o mesmo objeto encontrado
# O quarto parametro da NMSBoxes é para a supressão do objeto encontrado
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_expect, 0.4)
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    # Se o idx do objeto foi retornado no NMSBoxes o objeto é contornado
    if i in indexes:
        x, y, w, h = boxes[i]
        confidence = "(" + str(round((confidences[class_ids[i]]*100), 2)) + "%)"
        label = str(classes[class_ids[i]] + confidence)
        color = colors[class_ids[i]]

        # Monta o retangulo ao redor do objeto
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

# Mostra a imagem depois de processada
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
