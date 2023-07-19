import numpy as np
import tensorflow as tf
from tensorflow.keras import models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

# !! Modify this in the correct order
commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']

# loaded_model = models.load_model("saved_model")
loaded_model = tf.saved_model.load("saved_model")


def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    # Add this line to reshape
    spec = tf.convert_to_tensor(spec, dtype=tf.float32)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break