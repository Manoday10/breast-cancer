from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from news import NEWS_API_KEY
import requests
from werkzeug.utils import secure_filename
from utils import generate_insights
from models import predict_cancer, predict_cancer_from_csv
import markdown  # Import markdown library
from scraper import scrape_doctor_data 
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry backend for non-GUI operations
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import logging
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')  # Use environment variable for secret key

logging.basicConfig(level=logging.INFO)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# ALLOWED_CSV_EXTENSIONS = {'csv'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def ensure_upload_folder_exists():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

def fetch_doctor_data(location, specialization):
    return scrape_doctor_data(location, specialization)

def create_visualization(data):
    plt.figure(figsize=(8, 5))
    labels, values = zip(*data.items())
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Sample Visualization')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode()

def get_gemini_response(question):
    """Get response from the Gemini chatbot."""
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    
    response_text = ""
    for chunk in response:
        response_text += chunk.text + "\n"
    
    return response_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')



@app.route('/upload_image', methods=['POST'])
def upload_image():
    ensure_upload_folder_exists()
    
    if 'image' not in request.files:
        flash('No image file uploaded')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            prediction = predict_cancer(image_path)

            insights_md = generate_insights(predicted_label=prediction, file_type="Image")
            insights_html = markdown.markdown(insights_md)

            session['prediction'] = prediction
            session['insights'] = insights_html
            session['file_name'] = filename 

            # Pass the prediction to the Gemini chatbot
            questions = (
    f"A patient has been diagnosed with {prediction}. What are the recommended treatment options?",
    f"A patient has been diagnosed with {prediction}. What is the prognosis?",
    f"A patient has been diagnosed with {prediction}. What are the common symptoms?",
    f"A patient has been diagnosed with {prediction}. What are the risk factors for developing it?",
    f"A patient has been diagnosed with {prediction}. What are the stages and how do they affect treatment?",
    f"A patient has been diagnosed with {prediction}. What are the best follow-up care practices?",
    f"A patient has been diagnosed with {prediction}. What lifestyle changes can they make to improve outcomes?",
    f"A patient has been diagnosed with {prediction}. What further diagnostic tests might be needed?",
    f"A patient has been diagnosed with {prediction}. How does smoking affect their risk?",
    f"A patient has been diagnosed with {prediction}. What role does diet play in prevention?",
    f"A patient has been diagnosed with {prediction}. How does early detection influence treatment and outcomes?",
    f"A patient has been diagnosed with {prediction}. What is the difference between this and other types of cancer?",
    f"A patient has been diagnosed with {prediction}. Should genetic testing be considered?",
    f"A patient has been diagnosed with {prediction}. What support resources are available?",
    f"A patient has been diagnosed with {prediction}. Are there any alternative therapies that could complement treatment?",
    
    # New questions
    f"Given a diagnosis of {prediction}, what are the most effective treatment protocols currently available?",
    f"What are the latest advancements in treatment options for patients diagnosed with {prediction}?",
    f"What are the potential side effects of the recommended treatments for {prediction}, and how can they be managed?",
    f"How can patients cope with the side effects of treatments related to {prediction}?",
    f"Are there any ongoing clinical trials for {prediction} that patients might consider participating in?",
    f"What criteria should a patient meet to qualify for clinical trials related to {prediction}?",
    f"What psychological support options are available for patients diagnosed with {prediction}?",
    f"How can families support a loved one who has been diagnosed with {prediction}?",
    f"What is the long-term outlook for patients diagnosed with {prediction}, and what factors influence their survival rates?",
    f"How can patients diagnosed with {prediction} maintain their quality of life during and after treatment?",
    f"What dietary recommendations are suggested for patients undergoing treatment for {prediction}?",
    f"How can nutrition play a role in recovery for patients with {prediction}?",
    f"What are the implications of genetic factors for patients diagnosed with {prediction}?",
    f"Should family members of a patient diagnosed with {prediction} undergo genetic testing?",
    f"What integrative or complementary therapies are recommended alongside traditional treatments for {prediction}?",
    f"How can holistic approaches contribute to the well-being of patients diagnosed with {prediction}?",
    f"What does a typical follow-up care plan look like for someone diagnosed with {prediction}?",
    f"How often should patients with {prediction} schedule follow-up appointments, and what tests should be included?",
    f"What resources are available to help patients manage the financial burden associated with treatment for {prediction}?",
    f"How can patients ensure their treatment for {prediction} is covered by insurance?",
    
    "User can ask questions related to this, so you should be able to answer this type of questions."
)

            chatbot_response = get_gemini_response(questions)

            sample_data = {'Category A': 10, 'Category B': 15, 'Category C': 7}
            plot_image = create_visualization(sample_data)

            return render_template(
                'results.html',
                prediction=prediction,
                file_type='Image',
                file_name=filename,
                insights=insights_html,
                chatbot_response=chatbot_response,  # Include chatbot response
                plot_image=plot_image  # Pass plot image to the template
            )
        except Exception as e:
            logging.error(f'Error processing image: {str(e)}')
            flash('An error occurred during processing the image. Please try again.')
            return redirect('/')
    
    else:
        flash('Invalid image file type')
        return redirect('/')

@app.route('/doctors', methods=['GET', 'POST'])
def doctors():
    doctor_data = []
    if request.method == 'POST':
        location = request.form['location']
        specialization = request.form['specialization']
        
        try:
            doctor_data = fetch_doctor_data(location, specialization)
        except Exception as e:
            logging.error(f'Error fetching doctor data: {str(e)}')
            flash('An error occurred while fetching doctor data. Please try again.')

    return render_template('doctors.html', doctors=doctor_data)

@app.route('/news', methods=['GET', 'POST'])
def news():
    query = "CANCER" 

    if request.method == 'POST':
        query = request.form.get('query')  

    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get('articles', [])
    except requests.RequestException as e:
        logging.error(f'Error fetching news articles: {str(e)}')
        articles = []  # Fallback to an empty list if an error occurs
        flash('An error occurred while fetching news articles. Please try again.')

    return render_template('news.html', articles=articles)

@app.route('/charts', methods=['GET', 'POST'])
def charts():
    asr_df = pd.read_csv('data/dataset-asr-inc-both-sexes-in-2022-world.csv')
    mortality_df = pd.read_csv('data/dataset-asr-mort-both-sexes-in-2022-world.csv')

    asr_data = asr_df[['Label', 'ASR (World)']].dropna()
    mortality_data = mortality_df[['Label', 'ASR (World)']].dropna()

    prevalent_all_df = pd.read_csv('data/dataset-estimated-number-of-prevalent-cases-1-year-both-sexes-in-2022-all-cancers.csv')
    prevalent_all_df = prevalent_all_df.drop_duplicates(subset='Label')

    pie_all_labels = prevalent_all_df['Label'].tolist()
    pie_all_data = prevalent_all_df['Total'].tolist()

    pie_all_colors = [
        'rgba(255, 99, 132, 0.6)',    # Red
        'rgba(54, 162, 235, 0.6)',    # Blue
        'rgba(255, 206, 86, 0.6)',    # Yellow
        'rgba(75, 192, 192, 0.6)',    # Green
        'rgba(153, 102, 255, 0.6)',   # Purple
        'rgba(255, 159, 64, 0.6)',    # Orange
        'rgba(199, 199, 199, 0.6)',   # Grey
        'rgba(83, 102, 255, 0.6)',    # Indigo
        'rgba(255, 99, 71, 0.6)',     # Tomato
        'rgba(60, 179, 113, 0.6)'     # Medium Sea Green
    ]

    if len(pie_all_labels) > len(pie_all_colors):
        factor = (len(pie_all_labels) // len(pie_all_colors)) + 1
        pie_all_colors = (pie_all_colors * factor)[:len(pie_all_labels)]

    prevalent_cancer_df = pd.read_csv('data/dataset-estimated-number-of-prevalent-cases-1-year-both-sexes-in-2022-continents.csv')
    prevalent_cancer_df = prevalent_cancer_df.drop_duplicates(subset='Label')

    pie_cancer_labels = prevalent_cancer_df['Label'].tolist()
    pie_cancer_data = prevalent_cancer_df['Total'].tolist()

    pie_cancer_colors = [
        'rgba(255, 99, 132, 0.6)',    # Red
        'rgba(54, 162, 235, 0.6)',    # Blue
        'rgba(255, 206, 86, 0.6)',    # Yellow
        'rgba(75, 192, 192, 0.6)',    # Green
        'rgba(153, 102, 255, 0.6)',   # Purple
        'rgba(255, 159, 64, 0.6)',    # Orange
        'rgba(199, 199, 199, 0.6)',   # Grey
        'rgba(83, 102, 255, 0.6)',    # Indigo
        'rgba(255, 99, 71, 0.6)',     # Tomato
        'rgba(60, 179, 113, 0.6)'     # Medium Sea Green
    ]

    if len(pie_cancer_labels) > len(pie_cancer_colors):
        factor = (len(pie_cancer_labels) // len(pie_cancer_colors)) + 1
        pie_cancer_colors = (pie_cancer_colors * factor)[:len(pie_cancer_labels)]

    asr_chart_data = asr_data.to_dict(orient='records')
    mortality_chart_data = mortality_data.to_dict(orient='records')

    scatter_df = pd.read_csv("data/dataset-mortality-scatter.csv")
    scatter_data = scatter_df[['x', 'y']].dropna().to_dict(orient='records')

    return render_template(
        'charts.html',
        pie_all_labels=pie_all_labels,
        pie_all_data=pie_all_data,
        pie_all_colors=pie_all_colors,
        pie_cancer_labels=pie_cancer_labels,
        pie_cancer_data=pie_cancer_data,
        pie_cancer_colors=pie_cancer_colors,
        asr_chart_data=asr_chart_data,
        mortality_chart_data=mortality_chart_data,
        scatter_data=scatter_data,
    )

@app.route('/ask_chatbot', methods=['POST'])
def ask_chatbot():
    user_question = request.form['user_question']
     # Get previous prediction and insights from session
    previous_prediction = session.get('prediction', '')
    previous_insights = session.get('insights', '')

    # Optionally include previous prediction and insights in the question context
    full_question = f"Based on the prediction '{previous_prediction}' and insights: {previous_insights}, {user_question}"

    chatbot_response = get_gemini_response(full_question)
    chatbot_response_og = markdown.markdown(chatbot_response)

    return render_template(
        'results.html',
        prediction=previous_prediction,
        file_type='image',  # Adjust based on your implementation
        file_name=session.get('file_name', ''),
        insights= previous_insights,  # Include the generated insights
        chatbot_response=chatbot_response_og
    )


if __name__ == '__main__':
    app.run(debug=True)
