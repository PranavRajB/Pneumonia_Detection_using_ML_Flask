import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Define route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle image upload and perform prediction
        image_file = request.files['imagefile']
        if image_file:
            # Create the uploads directory if it doesn't exist
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            
            # Save the uploaded image to the uploads directory
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)
            
            # Load the pickled model architecture
            architecture_path = 'C:/Users/Ramkrishna/model_architecture.pkl'
            with open(architecture_path, 'rb') as f:
                loaded_model_architecture = pickle.load(f)
            
            # Load the pickled model weights
            weights_path = 'C:/Users/Ramkrishna/model_weights.pkl'
            with open(weights_path, 'rb') as f:
                loaded_model_weights = pickle.load(f)
            
            # Create the model using the loaded architecture
            loaded_model = tf.keras.models.model_from_json(loaded_model_architecture)
            
            # Set the loaded weights to the model
            loaded_model.set_weights(loaded_model_weights)
            
            # Load and preprocess the image
            image = Image.open(image_path).resize((256, 256)).convert('L')
            input_image = np.array(image)
            input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
            
            # Make predictions
            predictions = loaded_model.predict(input_image)
            predicted_class_index = np.argmax(predictions)
            
            # Get the predicted class label
            class_names = ['Normal', 'Pneumonia', ...]  # Replace with your class labels
            predicted_class_label = class_names[predicted_class_index]
            
            # Return the prediction as a response
            return predicted_class_label
    
    # Render the home page template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
