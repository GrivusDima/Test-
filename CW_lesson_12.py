import cv2

net = cv2.dnn.readNetFromCaffe("MobileNet/mobilenet_v2_deploy.prototxt", "MobileNet/mobilenet_v2.caffemodel")
classes = []

with open("MobileNet/synset.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image = cv2.imread('images/birdie.png')
blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))


net.setInput(blob)
preds = net.forward()

index = preds[0].argmax()

label = classes[index] if index < len(classes) else "unknown"

conf = float(preds[0][index]) * 100

print(f'Class: {label}')
print(f'Probality: {round(conf, 2)}%')

text = label + ': ' + str(int(conf)) + '%'
cv2.putText(image, text, (10, 30), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

cv2.imshow('la resultato', image)
cv2.waitKey(0)
cv2.destroyAllWindows()