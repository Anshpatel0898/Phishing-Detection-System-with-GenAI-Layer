from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user 
import joblib
from link_analysis import analyze_links
from models import db, User, PasswordResetToken
from datetime import datetime, timedelta
import secrets
import os
from werkzeug.security import generate_password_hash, check_password_hash
from email_validator import validate_email, EmailNotValidError
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from flask_mail import Mail, Message

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your-email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your-app-password')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', 'your-email@gmail.com')

# Initialize extensions
db.init_app(app)
mail = Mail(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Create database tables
with app.app_context():
    db.create_all()

# Load the trained model and scaler
model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

def extract_features(text):
    # Initialize feature array (57 features as in Spambase)
    features = np.zeros(57)
    
    # Convert text to lowercase for analysis
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words) if len(words) > 0 else 1
    total_chars = len(text) if len(text) > 0 else 1
    
    # Enhanced phishing indicators with weights
    phishing_indicators = {
        'urgent': {'words': ['urgent', 'immediately', 'asap', 'right away', 'hurry', 'limited time', 'expires soon'], 'weight': 0.3},
        'threat': {'words': ['suspended', 'closed', 'terminated', 'expired', 'blocked', 'security alert', 'unauthorized'], 'weight': 0.3},
        'request': {'words': ['verify', 'confirm', 'update', 'validate', 'authenticate', 'renew', 'reactivate'], 'weight': 0.2},
        'financial': {'words': ['account', 'bank', 'payment', 'invoice', 'transaction', 'credit card', 'paypal'], 'weight': 0.2},
        'personal': {'words': ['password', 'login', 'credentials', 'username', 'security', 'ssn', 'social security'], 'weight': 0.3},
        'generic': {'words': ['dear customer', 'dear user', 'dear member', 'valued customer', 'dear account holder'], 'weight': 0.1}
    }
    
    # Calculate phishing score based on indicators
    phishing_score = 0
    for category, data in phishing_indicators.items():
        for word in data['words']:
            if word in text_lower:
                phishing_score += data['weight']
    
    # Word frequencies (features 0-47)
    spam_words = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 
                 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 
                 'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george', 
                 '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 
                 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference']
    
    for i, word in enumerate(spam_words[:48]):
        features[i] = words.count(word) / total_words
    
    # Add phishing indicator scores
    features[0] = phishing_score  # Use first feature for phishing score
    
    # Character frequencies (features 48-54)
    features[48] = text.count(';') / total_chars
    features[49] = text.count('(') / total_chars
    features[50] = text.count('[') / total_chars
    features[51] = text.count('!') / total_chars
    features[52] = text.count('$') / total_chars
    features[53] = text.count('#') / total_chars
    
    # Capital letters (54)
    features[54] = sum(1 for c in text if c.isupper()) / total_chars
    
    # Word length statistics (55-56)
    if len(words) > 0:
        features[55] = sum(len(word) for word in words) / len(words)
        features[56] = max(len(word) for word in words)
    
    return features

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            flash('Please check your login details and try again.', 'danger')
            return redirect(url_for('login'))
        
        user.last_login = datetime.utcnow()
        db.session.commit()
        login_user(user, remember=remember)
        return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        try:
            validate_email(email)
        except EmailNotValidError:
            flash('Please enter a valid email address.', 'danger')
            return redirect(url_for('signup'))
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists.', 'danger')
            return redirect(url_for('signup'))
        
        new_user = User(
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate reset token
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(hours=1)
            
            reset_token = PasswordResetToken(
                user_id=user.id,
                token=token,
                expires_at=expires_at
            )
            
            db.session.add(reset_token)
            db.session.commit()
            
            # Send password reset email
            reset_url = url_for('reset_password', token=token, _external=True)
            msg = Message(
                'Password Reset Request - Phishing Detection System',
                recipients=[email]
            )
            msg.body = f'''
            Hello {user.first_name},

            You have requested to reset your password for the Phishing Detection System.
            Please click the following link to reset your password:

            {reset_url}

            This link will expire in 1 hour.

            If you did not request this password reset, please ignore this email.

            Best regards,
            Phishing Detection System Team
            '''
            try:
                mail.send(msg)
                flash('Password reset instructions have been sent to your email.', 'success')
            except Exception as e:
                app.logger.error(f"Failed to send email: {str(e)}")
                flash('Failed to send password reset email. Please try again later.', 'danger')
        else:
            # Don't reveal whether the email exists or not
            flash('If an account exists with that email, you will receive password reset instructions.', 'success')
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    reset_token = PasswordResetToken.query.filter_by(token=token).first()
    
    if not reset_token or reset_token.is_used or reset_token.expires_at < datetime.utcnow():
        flash('Invalid or expired reset token.', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        user = User.query.get(reset_token.user_id)
        
        user.set_password(password)
        reset_token.is_used = True
        db.session.commit()
        
        flash('Your password has been reset successfully!', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        email_text = request.form.get('email_content', '')
        if not email_text:
            return jsonify({
                "error": "Please provide email content to analyze",
                "details": "Email content is empty"
            }), 400

        # Extract features from the email text
        features = extract_features(email_text)
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        max_proba = max(proba)

        # Convert prediction to human-readable format
        prediction_text = 'phishing' if prediction == 1 else 'legitimate'
        
        # Calculate risk level based on probability and additional factors
        risk_score = max_proba
        if prediction == 1:  # If predicted as phishing
            # Increase risk score based on suspicious elements
            link_analysis = analyze_links(email_text)
            if link_analysis['found']:
                risk_score = min(1.0, risk_score + 0.2)  # Increase risk if suspicious links found
            
            # Check for common phishing indicators
            phishing_indicators = ['urgent', 'verify', 'account', 'password', 'suspended']
            for indicator in phishing_indicators:
                if indicator in email_text.lower():
                    risk_score = min(1.0, risk_score + 0.1)
        
        # Determine risk level
        if risk_score > 0.8:
            risk_level = 'high'
        elif risk_score > 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Get suspicious elements
        suspicious_elements = []
        if prediction == 1:
            for category, data in phishing_indicators.items():
                for word in data['words']:
                    if word in email_text.lower():
                        suspicious_elements.append(f"Contains {category} indicator: '{word}'")

        response = {
            'prediction': prediction_text,
            'confidence': f"{max_proba:.2%}",
            'risk_level': risk_level,
            'suspicious_elements': suspicious_elements,
            'report_phishing_url': "https://safebrowsing.google.com/safebrowsing/report_phish/?hl=en"
        }

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in predict route: {str(e)}")
        return jsonify({
            "error": "An error occurred while analyzing the email",
            "details": str(e)
        }), 500

def generate_pdf_report(email_text, prediction, confidence, risk_level, suspicious_elements):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom style for the report
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20
    )
    
    # Create the PDF content
    content = []
    
    # Title
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.blue
    )
    content.append(Paragraph("Phishing Detection Report", title_style))
    
    # Analysis Results
    content.append(Paragraph("Analysis Results:", styles['Heading2']))
    content.append(Paragraph(f"Prediction: {prediction}", custom_style))
    content.append(Paragraph(f"Confidence: {confidence}", custom_style))
    content.append(Paragraph(f"Risk Level: {risk_level}", custom_style))
    
    # Suspicious Elements
    if suspicious_elements:
        content.append(Paragraph("Suspicious Elements Found:", styles['Heading2']))
        for element in suspicious_elements:
            content.append(Paragraph(f"• {element}", custom_style))
    
    # Email Content
    content.append(Paragraph("Analyzed Email Content:", styles['Heading2']))
    content.append(Paragraph(email_text, custom_style))
    
    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

@app.route('/download-report', methods=['POST'])
@login_required
def download_report():
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        prediction = data.get('prediction', '')
        confidence = data.get('confidence', '')
        risk_level = data.get('risk_level', '')
        suspicious_elements = data.get('suspicious_elements', [])
        
        pdf_buffer = generate_pdf_report(
            email_text, 
            prediction, 
            confidence, 
            risk_level, 
            suspicious_elements
        )
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='phishing_analysis_report.pdf'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
