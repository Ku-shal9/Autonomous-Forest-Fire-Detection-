from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='forest_fire_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expect JSON with image data (224x224x3 array)
        data = request.get_json()
        image = np.array(data['image'], dtype=np.float32)
        image = image / 255.0  # Normalize
        image = image.reshape(1, 224, 224, 3)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        result = {'fire_probability': float(prediction[0][0])}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)