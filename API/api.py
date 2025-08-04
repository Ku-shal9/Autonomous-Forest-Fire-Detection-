from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import uvicorn

app = FastAPI()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='Model/forest_fire_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post('/predict')
async def predict(data: dict):
    try:
        # Expect JSON with image data (224x224x3 array)
        image = np.array(data['image'], dtype=np.float32)
        image = image / 255.0  # Normalize
        image = image.reshape(1, 224, 224, 3)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        result = {'fire_probability': float(prediction[0][0])}
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/health')
async def health():
    return {'status': 'API is running'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)