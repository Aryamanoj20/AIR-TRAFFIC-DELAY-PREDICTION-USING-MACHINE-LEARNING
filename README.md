# Flight Delay Prediction Using Machine Learning (Random Forest)

## Project Overview
This project predicts flight delays for **11 busy origin airports in the United States** using historical flight and weather data. The model is built using a **Random Forest Regressor**, which forecasts delays based on flight parameters such as airline, departure/arrival times, origin/destination airports, and weather conditions.  

The goal is to help airlines, passengers, and logistics planners manage flight schedules effectively by predicting potential delays in advance.

---

## Origin Airports Included
1. Hartsfield–Jackson Atlanta International Airport (ATL)  
2. Los Angeles International Airport (LAX)  
3. O'Hare International Airport (ORD)  
4. Dallas/Fort Worth International Airport (DFW)  
5. Denver International Airport (DEN)  
6. John F. Kennedy International Airport (JFK)  
7. San Francisco International Airport (SFO)  
8. Seattle-Tacoma International Airport (SEA)  
9. McCarran International Airport (LAS)  
10. Charlotte Douglas International Airport (CLT)  
11. Miami International Airport (MIA)  

---

## Project Features
- Predict flight delays using **Random Forest Regressor**.  
- Focused on **11 major US origin airports**.  
- Interactive **Streamlit app** for real-time predictions.  
- Modular code for training, prediction, and deployment.  
- Accepts user input for flights and provides estimated delays.  

---

## Project Structure
Flight-Delay-Prediction/
│
├─ training_model.ipynb      # VS Code code to train the Random Forest model
├─ prediction_code.py        # Script to make predictions using the trained model
├─ app.py                    # Streamlit app for interactive prediction
├─ README.md                 # Project description

## 🎯 Prediction Output
The model predicts the **delay in minutes** for a given flight.

- **Positive value (e.g., 15)** → Flight is delayed by 15 minutes.  
- **Zero (0)** → Flight is on time or arrived early (negative values are also considered as 0).


---

## Dataset
The dataset can be downloaded here:  
[Download processed_flights.csv](https://drive.google.com/file/d/16F67103nt6_NQ7o2QiFq_JBLQoLKHAAs/view?usp=drive_link)
