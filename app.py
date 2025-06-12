from flask import Flask, request, jsonify
import librosa, numpy as np, os, soundfile as sf
import tensorflow as tf

app = Flask(__name__)

# Load model
interpreter = tf.lite.Interpreter(model_path="emotion_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label sesuai urutan training model
labels = ['jijik', 'kecewa', 'netral', 'senang', 'terkejut']

def extract_mel(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < 128:
        log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])), mode='constant')
    else:
        log_mel = log_mel[:, :128]
    return log_mel[..., np.newaxis].astype(np.float32)  # [128, 128, 1]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "temp.wav"
    file.save(file_path)

    mel_input = extract_mel(file_path)
    mel_input = np.expand_dims(mel_input, axis=0)  # [1, 128, 128, 1]

    interpreter.set_tensor(input_details[0]['index'], mel_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_label = labels[np.argmax(output)]
    confidence = float(np.max(output))


    os.remove(file_path)

    
    return jsonify({
        "label": predicted_label,
        "confidence": float(np.max(output))
    })

if __name__ == '__main__':
    app.run(debug=True)
