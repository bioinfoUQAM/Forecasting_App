This is an interactive time series forecasting application built with Gradio. It allows you to train models and make predictions on various time series datasets through a simple web interface.

✨ Features

    Upload any time series dataset in CSV format

    Select forecasting models (ARIMA, N-BEATS and Bi-iGRU)

    Train interactively with adjustable parameters

    Visualize predictions in real time

🚀 How to Run

1 - Create and activate a virtual environment:

    python -m venv venv 
    source venv/bin/activate 
    
    venv\Scripts\activate # On Windows


2 - Install dependencies:

    pip install -r requirements.txt



3 - Launch the app:

    python app.py


4 - Open your browser at: http://127.0.0.1:7860


---------------------------------------------------------
To use it in Docker:

# Build the image (run in repo root)
docker build -t forecasting-app .

# Run container (mapping port 7860 on host to port 7860 in container)
docker run -p 7860:7860 forecasting-app
