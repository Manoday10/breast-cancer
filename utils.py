# utils.py

import  google.generativeai as genai
from google.generativeai import GenerativeModel, types
from dotenv import load_dotenv
import os
import csv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_insights(predicted_label, file_type, model_name="gemini-1.5-flash"):
    """Generate detailed medical insights based on the predicted label from the breast cancer detection model."""
    
    default_prompt = (
        f"As a medical expert specializing in oncology, you have detected {predicted_label} based on the "
        f"provided {file_type} data (hematoxylin and eosin-stained breast biopsy images). "
        f"The classification system used leverages Convolutional Neural Networks (CNNs) to categorize tissue "
        f"into four types: normal tissue, benign lesion, in situ carcinoma, and invasive carcinoma. "
        f"The system also supports a binary classification of carcinoma and non-carcinoma with high sensitivity.\n\n"
        
        f"Please provide the following in your response:\n\n"
        
        f"1. **Detailed Explanation**: Briefly explain what {predicted_label} is, the characteristics of "
        f"the detected tissue type (e.g., normal, benign, in situ carcinoma, invasive carcinoma), and "
        f"the significance of CNN-based detection in identifying cancerous and non-cancerous regions.\n"
        
        f"2. **Stage and Risk Factors**: Discuss potential stages of breast cancer (if applicable), "
        f"and elaborate on key risk factors. Mention how CNN architecture helps in differentiating tissue types, "
        f"including individual nuclei and overall tissue organization.\n"
        
        f"3. **Immediate Next Steps**: Suggest further diagnostic tests to confirm the classification, "
        f"such as biopsies, molecular testing, or additional imaging techniques. Highlight how early detection "
        f"through histology images impacts diagnostic accuracy and treatment planning.\n"
        
        f"4. **Treatment Options**: Outline the treatment options available based on the classification "
        f"of the biopsy (e.g., surgery, radiation, chemotherapy, targeted therapy, immunotherapy). Discuss early intervention "
        f"for in situ carcinoma or more aggressive treatments for invasive carcinoma if applicable.\n"
        
        f"5. **Lifestyle Adjustments and Prevention**: Recommend lifestyle changes or preventive measures "
        f"that the patient can take to improve health outcomes and reduce recurrence risk, especially for benign or early-stage cases.\n"
        
        f"6. **Prognosis and Patient Care**: Provide an overview of the prognosis based on the type and stage "
        f"of the tissue classification, discussing how early intervention, continuous monitoring, and follow-ups "
        f"are vital for patient care and improved long-term outcomes."
    )

    model = GenerativeModel(model_name)
    
    response_stream = model.generate_content(
        [default_prompt],
        generation_config=types.GenerationConfig(temperature=0.7),
        stream=True
    )
    
    insights_output = ""
    for message in response_stream:
        insights_output += message.text
    
    response_stream.resolve()
    return insights_output