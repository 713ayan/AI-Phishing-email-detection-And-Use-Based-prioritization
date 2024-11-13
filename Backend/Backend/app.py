from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import google.generativeai as genai
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import imaplib
import email
import yaml
from datetime import datetime
from urllib.parse import urlparse

nltk.download('punkt')
nltk.download('stopwords')

with open("credentials.yml") as f:
    creds = yaml.safe_load(f)

app = Flask(__name__)
CORS(app)

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

genai.configure(api_key=creds["genai_api_key"])

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def calculate_priority(subject, body, sender):
    priority_score = 0
    urgent_keywords = ['urgent', 'important', 'critical', 'asap', 'deadline']
    formal_salutations = ['dear', 'to whom it may concern', 'hello', 'hi']
    placement_related_keywords = ['placement', 'hiring', 'pre-placement', 'campus', 'interaction', 'company visit', 'recruitment']
    college_notice_keywords = ['notice', 'announcement', 'event', 'seminar', 'workshop', 'invitation', 'schedule', 'exam']

    if any(keyword in subject.lower() for keyword in urgent_keywords):
        priority_score += 30

    if any(salutation in body.lower()[:50] for salutation in formal_salutations):
        priority_score += 10

    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}', body):
        priority_score += 15

    priority_score += min(body.count('?') * 5, 20)

    if re.search(r'@(sitpune.edu.in)$', sender):
        priority_score += 50  # Ensures highest priority for emails from sitpune.edu.in
    elif re.search(r'@(ac|edu|gov|org)$', sender):
        priority_score += 20
    elif re.search(r'@(gmail|yahoo|hotmail|outlook)\.com$', sender):
        priority_score += 5

    tokens = preprocess_text(body)
    
    if any(keyword in tokens for keyword in placement_related_keywords):
        priority_score += 30
    elif any(keyword in tokens for keyword in college_notice_keywords):
        priority_score += 20

    important_words = ['meeting', 'report', 'project', 'deadline', 'interview', 'application', 'interaction', 'hiring', 'pre-placement', 'venue', 'eligible']
    priority_score += sum(5 for word in tokens if word in important_words)

    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    gemini_prompt = f"Analyze this email content: {body}. Determine if it's related to 'placement', 'college notice', or if it's of high priority based on urgency or important details."
    gemini_response = gemini_model.generate_content(gemini_prompt)
    gemini_priority = gemini_response.text.lower().strip()

    if 'high priority' in gemini_priority:
        priority_score += 20
    elif 'medium priority' in gemini_priority:
        priority_score += 10

    priority_score = min(priority_score, 100)

    return ('high', priority_score) if priority_score >= 80 else ('medium', priority_score) if priority_score >= 50 else ('low', priority_score)

def detect_phishing_links(body):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', body)
    suspicious_links = []
    for url in urls:
        parsed_url = urlparse(url)
        
        suspicious_tlds = ['.xyz', '.top', '.pw', '.tk', '.gq']
        if any(parsed_url.netloc.endswith(tld) for tld in suspicious_tlds):
            suspicious_links.append(url)
        
        url_shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']
        if any(shortener in parsed_url.netloc for shortener in url_shorteners):
            suspicious_links.append(url)
        
        if parsed_url.netloc.count('.') > 2:
            suspicious_links.append(url)
    
    return suspicious_links

def ai_predict_internal(subject, body, sender):
    try:
        if re.search(r'@(sitpune.edu.in)$', sender):
            priority, priority_score = calculate_priority(subject, body, sender)
            return {
                'ml_prediction': 'not spam',
                'ml_confidence': 1.0,
                'ai_prediction': 'not spam',
                'ai_explanation': 'Email is from sitpune.edu.in domain, automatically classified as not spam.',
                'final_prediction': 'not spam',
                'custom_message': 'This email is from sitpune.edu.in and will always be considered as not spam.',
                'priority': priority,
                'priority_score': priority_score,
                'suspicious_links': []  # No phishing check needed for trusted domain
            }
        
        input_features = vectorizer.transform([body])
        ml_prediction = model.predict(input_features)[0]
        ml_confidence = model.predict_proba(input_features)[0][ml_prediction]
        ml_result = 'not spam' if ml_prediction == 1 else 'spam'

        suspicious_links = detect_phishing_links(body)

        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"Analyze this email content: Subject: {subject}\nBody: {body}\n"
            "Determine if it's spam, or not spam. "
            "Consider the content, tone, and any suspicious elements. "
            "Respond with one of these categories: 'spam', or 'not spam'. "
            "Then, provide a brief explanation for your classification."
        )
        response = gemini_model.generate_content(prompt)
        ai_prediction, ai_explanation = response.text.split('\n', 1)
        ai_prediction = ai_prediction.replace("*","")
        if suspicious_links:
            final_prediction = 'phishing'
            custom_message = f"Potential phishing detected. Suspicious links found: {', '.join(suspicious_links)}"
        elif ai_prediction.lower() == 'phishing':
            final_prediction = 'phishing'
            custom_message = f"Potential phishing email detected. {ai_explanation.strip()}"
        elif ai_prediction.lower() == 'spam' or ml_result == 'spam':
            final_prediction = 'spam'
            custom_message = f"This email has been classified as spam. {ai_explanation.strip()}"
        else:
            final_prediction = 'not spam'
            custom_message = "This email appears to be not spam."

        priority, priority_score = calculate_priority(subject, body, sender)

        return {
            'ml_prediction': ml_result,
            'ml_confidence': float(ml_confidence),
            'ai_prediction': ai_prediction,
            'ai_explanation': ai_explanation.strip(),
            'final_prediction': final_prediction,
            'custom_message': custom_message,
            'priority': priority,
            'priority_score': priority_score,
            'suspicious_links': suspicious_links
        }

    except Exception as e:
        return {'error': str(e)}

@app.route('/api/emails', methods=['GET'])
def get_emails():
    imap_url = 'imap.gmail.com'
    my_mail = imaplib.IMAP4_SSL(imap_url)
    my_mail.login(creds["user"], creds["password"])
    my_mail.select('Inbox')

    today_date = datetime.now().strftime("%d-%b-%Y")
    _, data = my_mail.search(None, f'SINCE {today_date}')

    emails = []

    for num in data[0].split():
        _, msg_data = my_mail.fetch(num, '(RFC822)')
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)

        email_info = {
            "subject": clean_text(email_message['subject']),
            "from": clean_from(email_message['from']),
            "body": ""
        }
        
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    email_info["body"] = clean_text(part.get_payload(decode=True).decode())
        else:
            email_info["body"] = clean_text(email_message.get_payload(decode=True).decode())
        
        emails.append(email_info)

    my_mail.close()
    my_mail.logout()

    return jsonify(emails)

@app.route('/api/check-spam', methods=['POST'])
def check_spam():
    data = request.json
    subject = data.get('subject', '')
    body = data.get('body', '')
    sender = data.get('from', '')

    result = ai_predict_internal(subject, body, sender)
    return jsonify(result)


def clean_text(text):
    return re.sub(r'[\r\n\t]', '', text).strip()

def clean_from(from_text):
    if '<' in from_text:
        return from_text.split('<', 1)[1].strip('>')
    return from_text.strip()

if __name__ == '__main__':
    app.run(debug=True)