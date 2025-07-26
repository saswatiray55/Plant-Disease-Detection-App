# Import necessary libraries
from flask import Flask, request, render_template, url_for
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import requests # <-- NEW: For making API calls

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('plant_disease_model_v1.h5')

# Define class names and remedies (as before)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
]
remedies = {
    'Apple___Apple_scab': 'Remove and destroy infected leaves and fruit. Apply a fungicide containing myclobutanil or captan.',
    'Potato___Late_blight': 'Apply fungicides proactively, especially during cool, wet weather. Destroy infected plants to prevent spread.',
    'default': 'Ensure good air circulation, proper watering, and balanced fertilization.'
}

# --- NEW: Function to get weather data ---
def get_weather(city="Chandrakona"):
    # Replace 'YOUR_API_KEY' with the key you got from OpenWeatherMap
    API_KEY = 'c96d12d68a191a42bc57aaec527b28e8' 
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    # Construct the full URL with the city and your API key
    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        
        # Extract the relevant weather information
        weather = {
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'].title(),
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
        return weather
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather: {e}")
        return None


def preprocess_image(image_path, target_size=(224, 224)):
    """Loads and preprocesses an image for the model."""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    """Renders the home page."""
    weather_data = get_weather()
    return render_template('index.html', weather=weather_data)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload, prediction, and renders the result."""
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='Error: No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text='Error: No selected file')

    if file:
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        processed_image = preprocess_image(file_path)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        
        if 'healthy' in predicted_class_name.lower():
            remedy_text = "The plant appears to be healthy. Keep up the good work!"
        else:
            remedy_text = remedies.get(predicted_class_name, remedies['default'])
        
        display_image_path = f'uploads/{file.filename}'
        
        # --- NEW: Get weather data to pass to the result page ---
        weather_data = get_weather()

        return render_template('index.html', 
                               prediction_text=f'{predicted_class_name}',
                               image_path=display_image_path,
                               remedy_text=remedy_text,
                               weather=weather_data) # <-- Pass weather data

if __name__ == '__main__':
    app.run(debug=True)
