# Causal Estimator

This is a simple application to estimate causal effect based on a causal graph. It makes use of (streamlit)[https://streamlit.io/] for the UI and Microsoft's (DoWhy)[https://microsoft.github.io/dowhy/] for inference.

Usage:
- Build and run the docker image
- The application will be available at: http://localhost:8501
- At the UI you can import your csv file with the treatment, outcome and confounders features
- Build your causal graph at http://dagitty.net/dags.html and edit it to the format shown in the left panel
- Finally just run the analysis

<img src="images/ui_example.png">