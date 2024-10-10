import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from PIL import Image

modelSavePath = 'my_model3.h5'

# testImagePath = 'Invasive1.jpg'

classes = ["Benign", "InSitu", "Invasive", "Normal"]

def predict_cancer(img_path):
    """
    Predicts the cancer class from the given image path.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        str: Predicted cancer class.
    """
    model = load_model(modelSavePath, compile=False)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy')

    img = Image.open(img_path)
    z = np.asarray(img, dtype=np.int8)
    
    crops = []
    for i in range(3):
        for j in range(4):
            crop = z[512 * i:512 * (i + 1), 512 * j:512 * (j + 1), :]
            crops.append(crop)

    crops = np.array(crops, np.float16) / 255.0  # Normalize pixel values

    compProbs = [0] * len(classes)

    for crop in crops:
        x = np.expand_dims(crop, axis=0)
        softMaxPred = model.predict(x, verbose=0)
        
        z_exp = [np.math.exp(i) for i in softMaxPred[0]]
        sum_z_exp = sum(z_exp)
        probs = [(i / sum_z_exp) * 100 for i in z_exp]

        maxI = np.argmax(probs)
        compProbs[maxI] += probs[maxI]

    highest_prob_idx = np.argmax(compProbs)
    highest_class = classes[highest_prob_idx]

    return highest_class




def predict_cancer_from_csv(csv_path):
    """
    Predict cancer risk from CSV data using a trained ANN model.
    
    Parameters:
        csv_path (str): The path to the CSV file.
    
    Returns:
        str: The predicted cancer risk.
    """
    # Implement your CSV prediction logic here
    # Placeholder implementation
    return "Predicted cancer risk: High"
