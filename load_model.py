from tensorflow.keras.models import model_from_json

# Step 1: Load model architecture from JSON
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Step 2: Load weights into the model
model.load_weights("model_weights.h5")  # ⚠️ Change this if your weights file has a different name

print("✅ Model loaded successfully!")
