# predictive-maintainance-of-IOT-devices
python
CopyEdit
import Adafruit_DHT
import time

# Define sensor type and GPIO pin
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

while True:
    humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
    if humidity is not None and temperature is not None:
        print(f"Temperature: {temperature}°C, Humidity: {humidity}%")
    else:
        print("Sensor failure. Check connections.")
    time.sleep(2)



python
CopyEdit
import RPi.GPIO as GPIO
import time

VIBRATION_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(VIBRATION_PIN, GPIO.IN)

while True:
    vibration_detected = GPIO.input(VIBRATION_PIN)
    if vibration_detected == 1:
        print("⚠️ Vibration detected! Possible machine malfunction.")
    else:
        print("No vibration detected. Machine running smoothly.")
    time.sleep(1)

import time
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

while True:
    current_value = read_adc(0)  # Reading from Channel 0 of MCP3008
    print(f"Current Consumption: {current_value} mA")
    time.sleep(1)
python
CopyEdit
import serial

# Configure Serial Connection for ESP8266
esp8266 = serial.Serial('/dev/serial0', 115200, timeout=1)

def send_data_to_cloud(temperature, vibration, current):
    data = f"{temperature},{vibration},{current}"
    esp8266.write(data.encode())
    print(f"📡 Sent to Cloud: {data}")

while True:
    # Simulated sensor readings (replace with actual readings)
    temp = 35.0
    vib = 0
    curr = 500

    send_data_to_cloud(temp, vib, curr)
    time.sleep(5)


Control Motor & Fan Using L298N Motor Driver
import RPi.GPIO as GPIO
import time

MOTOR_PIN_1 = 22
MOTOR_PIN_2 = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_PIN_1, GPIO.OUT)
GPIO.setup(MOTOR_PIN_2, GPIO.OUT)

try:
    while True:
        print("Motor Running Forward")
        GPIO.output(MOTOR_PIN_1, GPIO.HIGH)
        GPIO.output(MOTOR_PIN_2, GPIO.LOW)
        time.sleep(5)

        print("Motor Running Backward")
        GPIO.output(MOTOR_PIN_1, GPIO.LOW)
        GPIO.output(MOTOR_PIN_2, GPIO.HIGH)
        time.sleep(5)

except KeyboardInterrupt:
    print("Stopping Motor")
    GPIO.cleanup()
import Adafruit_DHT
import RPi.GPIO as GPIO
import spidev
import serial
import time
import requests

# **1️⃣ Initialize GPIO & Components**
GPIO.setmode(GPIO.BCM)

# DHT11 (Temperature Sensor) Setup
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

# SW-420 (Vibration Sensor) Setup
VIBRATION_PIN = 17
GPIO.setup(VIBRATION_PIN, GPIO.IN)

# MCP3008 (Current Sensor via ADC)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# ESP8266 (Wi-Fi Module) Serial Setup
esp8266 = serial.Serial('/dev/serial0', 115200, timeout=1)

# L298N Motor Driver Setup
MOTOR_PIN_1 = 22
MOTOR_PIN_2 = 23
GPIO.setup(MOTOR_PIN_1, GPIO.OUT)
GPIO.setup(MOTOR_PIN_2, GPIO.OUT)

# **2️⃣ Function to Read All Sensor Data**
def get_sensor_data():
    # Read temperature & humidity
    humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
    if temperature is None:
        temperature = 0  # Handle sensor failure

    # Read vibration status
    vibration_detected = GPIO.input(VIBRATION_PIN)

    # Read current sensor data
    current_value = read_adc(0)  # Read from ADC Channel 0

    return temperature, humidity, vibration_detected, current_value

# **3️⃣ Function to Send Data to AI Model for Prediction**
def send_data_to_ai(temperature, vibration, current):
    # Send data to Flask AI API
    api_url = "http://127.0.0.1:5000/predict"  # Update with actual server if needed
    payload = {"temperature": temperature, "vibration": vibration, "current": current}
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        prediction = response.json()["Prediction"]
        print(f"🔍 AI Prediction: {prediction}")
    else:
        print("⚠️ Failed to get prediction from AI model.")

# **4️⃣ Function to Control Motor Based on AI Prediction**
def control_motor(prediction):
    if "failure predicted" in prediction.lower():
        print("⚠️ Motor OFF - Machine failure detected!")
        GPIO.output(MOTOR_PIN_1, GPIO.LOW)
        GPIO.output(MOTOR_PIN_2, GPIO.LOW)
    else:
        print("✅ Motor Running Normally")
        GPIO.output(MOTOR_PIN_1, GPIO.HIGH)
        GPIO.output(MOTOR_PIN_2, GPIO.LOW)

# **5️⃣ Main Loop: Continuously Read, Predict & Act**
try:
    while True:
        # Read sensor data
        temperature, humidity, vibration, current = get_sensor_data()
        print(f"📊 Sensor Data: Temp={temperature}°C, Vibration={vibration}, Current={current} mA")

        # Send sensor data to AI model
        send_data_to_ai(temperature, vibration, current)

        # Wait before next reading
        time.sleep(5)

except KeyboardInterrupt:
    print("❌ Stopping System")
    GPIO.cleanup()

python
CopyEdit
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sensor data (simulated)
data = pd.read_csv("iot_sensor_data.csv")

# Preprocess data
X = data[['temperature', 'vibration', 'current']]
y = data['failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
import joblib
joblib.dump(model, "predictive_maintenance_model.pkl")
python
CopyEdit
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("predictive_maintenance_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data["temperature"], data["vibration"], data["current"]]])
    prediction = model.predict(features)
    result = "Machine failure predicted!" if prediction[0] == 1 else "Machine is running normally."
    return jsonify({"Prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
2.	Test API using Postman: 
o	Send a POST request to http://127.0.0.1:5000/predict
o	JSON Input: 
json
CopyEdit
{
    "temperature": 75,
    "vibration": 4.5,
    "current": 12.3
}
o	Response: 
json
CopyEdit
{"Prediction": "Machine is running normally."}

