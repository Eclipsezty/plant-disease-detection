# Plant Disease Detection Android Application

This README provides an overview of the Android portion of the "Plant Disease Detection" project. The application is designed to detect plant diseases using a combination of image recognition and server-based deep learning algorithms. It includes features such as image capture, location tracking, and server communication.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [How It Works](#how-it-works)
6. [Future Enhancements](#future-enhancements)

---

## Overview
The Android application is part of a system designed to help farmers identify plant diseases. It works by:
- Allowing users to capture or upload images of plant leaves.
- Sending the image to a server for analysis.
- Receiving and displaying the disease detection results.

The application also retrieves the device's location and allows the user to configure the server IP and port.

---

## Features
1. **Image Capture and Upload**:
   - Capture photos using the device camera.
   - Upload existing photos from the gallery.
2. **Server Communication**:
   - Sends Base64-encoded images to a server.
   - Displays disease detection results from the server.
3. **Location Tracking**:
   - Retrieves the device’s current latitude and longitude.
4. **Configurable Network Settings**:
   - Allows users to input and save the server’s IP address and port.
5. **User-Friendly Interface**:
   - Splash screen, login page, and intuitive navigation through a menu.

---

## Project Structure
The project follows a modular structure with separate Activities for each feature:

```
.
├── app/
   ├── src/
       ├── main/
           ├── java/com/example/android/
               ├── DiseaseDetectActivity.java  # Main activity for disease detection
               ├── GetValue.java               # Activity for configuring server settings
               ├── LoginActivity.java           # Login screen activity
               ├── MenuActivity.java            # Main menu activity
               ├── SplashActivity.java          # Splash screen activity
           ├── res/
               ├── layout/                     # XML files for activity layouts
               ├── mipmap/                     # App icons and resources
           ├── AndroidManifest.xml            # App configuration file
   ├── build.gradle                           # Gradle build configuration
```

---

## Setup Instructions

### Prerequisites
1. **Android Studio**: Install the latest version of Android Studio.
2. **Dependencies**: Ensure the following libraries are included:
   - Google Play Services for location tracking.
   - Permissions handling (AppCompat).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Eclipsezty/plant-disease-detection.git
   ```
2. Open the project in Android Studio.
3. Configure the server IP and port:
   - Default IP: `192.168.0.102`
   - Default Port: `6666`
   - Update these values in `DiseaseDetectActivity.java` or use the "Network Configuration" feature in the app.
4. Build and run the project on an emulator or physical device.
5. Grant necessary permissions when prompted (Camera, Location, Phone State).

---

## How It Works
1. **User Interaction**:
   - The user captures or selects an image of a plant leaf.
   - The app retrieves the current location and sends the image and metadata to the server.

2. **Server Communication**:
   - Uses a socket connection to transmit data (image in Base64 format, location, and phone number).
   - Receives disease detection results from the server.

3. **Display Results**:
   - Displays the disease type and confidence score to the user.

4. **Activities Overview**:
   - **SplashActivity**: Displays a welcome screen for 3 seconds.
   - **LoginActivity**: Provides access to the main menu.
   - **MenuActivity**: Acts as a hub for navigation.
   - **DiseaseDetectActivity**: Handles image capture, processing, and server communication.
   - **GetValue**: Allows server settings to be configured.

---

## Future Enhancements
1. **Cloud Deployment**:
   - Allow server deployment on cloud platforms for global access.
2. **Improved UI**:
   - Add animations and improve layout designs.
3. **Model Integration**:
   - Allow switching between different disease detection models on the server.
4. **Offline Mode**:
   - Provide basic disease detection capabilities without a server.
5. **Data Analytics**:
   - Integrate data visualization for historical detection results.

---

## Contact
For questions or contributions, please contact:
- **Name**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [Your GitHub Profile]

---
