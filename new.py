import cv2
import numpy as np
import tensorflow as tf

loaded_model = tf.keras.models.load_model('model_checkpoint.h5')
print(loaded_model.summary())

labels = ['maryam', 'not me', 'shaheen']
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.resize(frame, (220, 220))

    image = np.array([image])

    prediction = loaded_model.predict(image)
    prob = prediction[0][np.argmax(prediction)]
    name = labels[np.argmax(prediction)]
    print(prediction)


    cv2.putText(frame, f' {name} Probability: {prob}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
