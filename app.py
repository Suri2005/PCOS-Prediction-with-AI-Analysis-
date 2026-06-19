import os
import json
import logging
import requests
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from io import BytesIO
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash

import shap
import qrcode
import plotly.express as px
import plotly.graph_objects as go
import plotly

# ReportLab imports for Certificate generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

from config import Config

# Setup Logger for API Logs
api_logger = logging.getLogger('api_transactions')
api_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('api_logs.txt')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
api_logger.addHandler(file_handler)

def log_api_transaction(api_key, input_data, prediction, confidence, execution_time_ms):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'api_key': api_key,
        'input_features': input_data,
        'prediction': prediction,
        'confidence': confidence,
        'execution_time_ms': execution_time_ms
    }
    api_logger.info(json.dumps(log_entry))


app = Flask(__name__)
app.config.from_object(Config)

# Ensure required directories exist
os.makedirs(os.path.join(app.root_path, 'uploads', 'certificates'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'database'), exist_ok=True)

# Initialize Flask extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# --------------------------------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------------------------------
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    patients = db.relationship('Patient', backref='user', lazy=True, cascade="all, delete-orphan")
    assessments = db.relationship('Assessment', backref='user', lazy=True, cascade="all, delete-orphan")

class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    age = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    assessments = db.relationship('Assessment', backref='patient', lazy=True, cascade="all, delete-orphan")

class Assessment(db.Model):
    __tablename__ = 'assessments'
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), default=lambda: str(uuid4()), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Clinical parameters
    cycle_length = db.Column(db.Float, nullable=False)
    fsh = db.Column(db.Float, nullable=False)
    lh = db.Column(db.Float, nullable=False)
    amh = db.Column(db.Float, nullable=False)
    testosterone = db.Column(db.Float, nullable=False)
    follicle_l = db.Column(db.Integer, nullable=False)
    follicle_r = db.Column(db.Integer, nullable=False)
    fast_food = db.Column(db.Integer, nullable=False)
    
    # 10 symptom severity sliders (1 to 5)
    acne = db.Column(db.Integer, nullable=False)
    hair_loss = db.Column(db.Integer, nullable=False)
    hirsutism = db.Column(db.Integer, nullable=False)
    weight_gain = db.Column(db.Integer, nullable=False)
    darkening = db.Column(db.Integer, nullable=False)
    cycle = db.Column(db.Integer, nullable=False)
    fatigue = db.Column(db.Integer, nullable=False)
    mood_swings = db.Column(db.Integer, nullable=False)
    headaches = db.Column(db.Integer, nullable=False)
    conceiving = db.Column(db.Integer, nullable=False)
    symptom_score = db.Column(db.Float, nullable=False)
    
    # Predictions
    risk_pct = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    pcos_detected = db.Column(db.Boolean, nullable=False)
    clinical_summary = db.Column(db.Text, nullable=False)
    
    # AI Report features
    gemini_summary = db.Column(db.Text, nullable=True)
    doctor_note = db.Column(db.Text, nullable=True)
    tags = db.Column(db.String(200), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --------------------------------------------------------------------------
# MACHINE LEARNING ENGINE LOADER
# --------------------------------------------------------------------------
model = None
scaler = None
model_loaded = False

def load_ml_components():
    global model, scaler, model_loaded
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'pcos_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            model_loaded = True
            print("Successfully loaded ensemble voting model and scaler.")
        else:
            print("Model files not found. Please run 'train_model.py' to generate model and scaler.")
            model_loaded = False
    except Exception as e:
        print(f"Error loading machine learning components: {e}")
        model_loaded = False

load_ml_components()

# --------------------------------------------------------------------------
# SHAP RISK ATTRIBUTION & FALLBACKS
# --------------------------------------------------------------------------
def get_shap_explanation(rf_model, scaler, sample_df, feature_names):
    try:
        # Check if shap is installed and imported
        explainer = shap.TreeExplainer(rf_model)
        sample_scaled = scaler.transform(sample_df)
        shap_vals = explainer.shap_values(sample_scaled)
        
        # Resolve class 1 (positive case) values
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif isinstance(shap_vals, np.ndarray):
            if len(shap_vals.shape) == 3:
                vals = shap_vals[1][0]
            elif len(shap_vals.shape) == 2:
                vals = shap_vals[0]
            else:
                vals = shap_vals
        else:
            vals = shap_vals[0]
            
        attributions = []
        for name, val in zip(feature_names, vals):
            attributions.append({'feature': name, 'value': float(val)})
            
    except Exception as e:
        print(f"SHAP calculations failed: {e}. Executing weighted contribution fallback.")
        attributions = []
        try:
            rf_importances = joblib.load('model/feature_importance.pkl')
        except Exception:
            rf_importances = {
                'Age': 0.04, 'BMI': 0.07, 'Cycle(R/I)': 0.12, 'FSH(mIU/mL)': 0.05, 'LH(mIU/mL)': 0.09,
                'AMH(ng/mL)': 0.15, 'Testosterone(ng/dL)': 0.14, 'Follicle No.(L)': 0.18, 'Follicle No.(R)': 0.16,
                'Skin_darkening': 0.08, 'hair_growth': 0.09, 'Weight_gain': 0.06, 'Cycle_length(days)': 0.05,
                'Fast_food': 0.02, 'symptom_score': 0.10
            }
            
        normal_baselines = {
            'Age': 25.0, 'BMI': 22.0, 'Cycle(R/I)': 0.0, 'FSH(mIU/mL)': 3.0, 'LH(mIU/mL)': 3.0,
            'AMH(ng/mL)': 2.0, 'Testosterone(ng/dL)': 30.0, 'Follicle No.(L)': 5.0, 'Follicle No.(R)': 5.0,
            'Skin_darkening': 0.0, 'hair_growth': 0.0, 'Weight_gain': 0.0, 'Cycle_length(days)': 28.0,
            'Fast_food': 0.0, 'symptom_score': 0.2
        }
        
        for name in feature_names:
            val = float(sample_df[name].iloc[0])
            base = normal_baselines.get(name, 0.0)
            importance = rf_importances.get(name, 0.05)
            
            if name == 'FSH(mIU/mL)':
                diff = (base - val) / 5.0
            elif name == 'Age':
                diff = 0.0
            else:
                diff = (val - base) / (base if base > 0 else 1.0)
                
            contribution = diff * importance
            attributions.append({'feature': name, 'value': contribution})
            
    # Partition factors into positive (risk up) and negative (risk down)
    attributions.sort(key=lambda x: x['value'], reverse=True)
    up_factors = [x for x in attributions if x['value'] > 0][:5]
    down_factors = sorted([x for x in attributions if x['value'] < 0], key=lambda x: x['value'])[:3]
    
    feature_display_names = {
        'Age': 'Patient Age',
        'BMI': 'Body Mass Index (BMI)',
        'Cycle(R/I)': 'Irregular Menstrual Cycle',
        'FSH(mIU/mL)': 'FSH Level',
        'LH(mIU/mL)': 'LH Level',
        'AMH(ng/mL)': 'AMH Hormone Level',
        'Testosterone(ng/dL)': 'Total Testosterone Level',
        'Follicle No.(L)': 'Left Ovary Follicles Count',
        'Follicle No.(R)': 'Right Ovary Follicles Count',
        'Skin_darkening': 'Skin Darkening Severity',
        'hair_growth': 'Hirsutism / Hairy Growth',
        'Weight_gain': 'Unexplained Weight Gain',
        'Cycle_length(days)': 'Cycle Duration',
        'Fast_food': 'Frequent Fast Food Intake',
        'symptom_score': 'Total Symptom Severity Index'
    }
    
    formatted_up = [{'name': feature_display_names.get(f['feature'], f['feature']), 'impact': round(f['value'] * 100, 1)} for f in up_factors]
    formatted_down = [{'name': feature_display_names.get(f['feature'], f['feature']), 'impact': round(abs(f['value']) * 100, 1)} for f in down_factors]
    
    return formatted_up, formatted_down

# --------------------------------------------------------------------------
# GOOGLE GEMINI CLINICAL REPORT WRITER
# --------------------------------------------------------------------------
def call_gemini_api(api_key, patient_data):
    if not api_key or api_key == "your_actual_gemini_api_key_here":
        return get_gemini_fallback(patient_data)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    prompt = f"""
    You are an expert reproductive endocrinologist. Analyze this PCOS patient screening:
    Patient Details:
    - Name: {patient_data['name']}
    - Age: {patient_data['age']}
    - BMI: {patient_data['bmi']}
    - PCOS Class: {'Detected (High Risk)' if patient_data['pcos_detected'] else 'Not Detected (Low Risk)'}
    - Risk Probability: {patient_data['risk_pct']}%
    - Key metrics: Left follicles={patient_data['follicle_l']}, Right follicles={patient_data['follicle_r']}, AMH={patient_data['amh']} ng/mL, Testosterone={patient_data['testosterone']} ng/dL, LH/FSH ratio={patient_data['lh_fsh_ratio']}.
    - Physical symptoms (1-5 severity): Acne={patient_data['acne']}, Hair loss={patient_data['hair_loss']}, Hirsutism={patient_data['hirsutism']}, Weight gain={patient_data['weight_gain']}, Skin darkening={patient_data['darkening']}, Menstrual irregularity={patient_data['cycle']}.
    
    Generate a JSON response containing three fields:
    1. "summary": A personalized, compassionate, and clinically accurate summary (2-3 sentences).
    2. "doctor_note": Direct lifestyle guidance and medical advice tailored to their profile (3-4 sentences). Emphasize endocrinology best practices, carbohydrate control, stress management, and physical training.
    3. "tags": A list of 3-4 short motivational tag strings (e.g. ["Hormonal Balance", "Active Healing", "Metabolic Health"]).
    
    Return ONLY valid raw JSON. Do not include markdown code block formatting. Do not wrap in ```json. Just raw JSON.
    """
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=12)
        if response.status_code == 200:
            res_json = response.json()
            text_response = res_json['contents'][0]['parts'][0]['text'].strip()
            # Strip markdown formatting if the model ignored request
            if text_response.startswith("```"):
                text_response = text_response.split("```")[1]
                if text_response.startswith("json"):
                    text_response = text_response[4:]
            data = json.loads(text_response.strip())
            return data
        else:
            print(f"Gemini API returned code {response.status_code}: {response.text}")
            return get_gemini_fallback(patient_data)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return get_gemini_fallback(patient_data)

def get_gemini_fallback(patient_data):
    name = patient_data['name']
    pcos = patient_data['pcos_detected']
    
    if pcos:
        summary = (
            f"Hello {name}, your risk screening results indicate biochemical and physical parameters "
            f"closely aligned with PCOS. Elevated indicators include an ovarian follicle count of "
            f"{patient_data['follicle_l'] + patient_data['follicle_r']} and a luteinizing hormone ratio of {patient_data['lh_fsh_ratio']}, "
            f"suggesting typical hyperandrogenic and polycystic ovarian features."
        )
        doctor_note = (
            "We recommend scheduling a formal consultation with a gynecologist or endocrinologist. "
            "To support hormone regulation, focus on insulin-sensitizing lifestyle adjustments: implement a "
            "low-glycemic index whole-food diet, integrate consistent resistance training, maintain good sleep hygiene, "
            "and discuss options like inositol supplementation with your physician."
        )
        tags = ["Androgen Support", "Low-GI Habits", "Ovarian Health", "Strength & Healing"]
    else:
        summary = (
            f"Hello {name}, your assessment indicates a healthy low-risk profile. Your androgenic hormones, "
            f"anti-müllerian hormone ({patient_data['amh']} ng/mL), and follicular activity are well within standard, "
            f"non-polycystic ranges."
        )
        doctor_note = (
            "Your metrics indicate excellent metabolic and endocrine balance. Continue prioritizing a "
            "nutrient-dense whole-food diet, regular cardiovascular and strength exercise, and active stress "
            "reduction strategies. Keep tracking your cycle regularly as a key indicator of continued health."
        )
        tags = ["Metabolic Fitness", "Endocrine Balance", "Cycle Harmony", "Optimal Wellness"]
        
    return {
        "summary": summary,
        "doctor_note": doctor_note,
        "tags": tags
    }

# --------------------------------------------------------------------------
# ROTATIONAL 7-DAY PERSONALIZED MEAL PLANNER
# --------------------------------------------------------------------------
MEAL_PLANS = {
    'A': {
        'title': 'High-Protein Low-Carb Caloric Deficit Plan',
        'subtitle': 'Tailored for PCOS Management with Weight Regulation (BMI ≥ 25)',
        'description': 'Designed to lower insulin resistance, reduce androgen levels, and support sustainable fat loss.',
        'days': [
            {'day': 'Monday', 'breakfast': 'Spinach & mushroom scramble with 2 eggs, avocado slice.', 'lunch': 'Grilled chicken breast over a large green salad with olive oil dressing.', 'snack': 'A handful of walnuts and pumpkin seeds.', 'dinner': 'Baked salmon with steamed broccoli and cauliflower mash.', 'fluids': '2.5L water, 1 cup spearmint tea.'},
            {'day': 'Tuesday', 'breakfast': 'Chia seed pudding with unsweetened almond milk and raspberries.', 'lunch': 'Turkey breast lettuce wraps with cucumber and avocado salsa.', 'snack': 'Plain Greek yogurt with cinnamon.', 'dinner': 'Grilled grass-fed beef sirloin with roasted asparagus.', 'fluids': '2.5L water, 1 cup green tea.'},
            {'day': 'Wednesday', 'breakfast': 'Tofu scramble with bell peppers, spinach, and turmeric.', 'lunch': 'Tuna salad with olive oil, celery, and mixed greens.', 'snack': 'Celery sticks with almond butter.', 'dinner': 'Baked chicken thighs with garlic roasted zucchini and green beans.', 'fluids': '2.5L water, 1 cup spearmint tea.'},
            {'day': 'Thursday', 'breakfast': '2 boiled eggs, half an avocado, and sliced cucumber.', 'lunch': 'Shrimp stir-fry with broccoli, snap peas, and sesame oil.', 'snack': 'Pumpkin seeds and a small piece of dark chocolate (85%+).', 'dinner': 'Turkey meatballs with zucchini noodles and marinara sauce.', 'fluids': '2.5L water, 1 cup ginger tea.'},
            {'day': 'Friday', 'breakfast': 'Whey protein shake with spinach, unsweetened almond milk, and flaxseeds.', 'lunch': 'Sliced roast beef over arugula, cherry tomatoes, and olive oil.', 'snack': '1 hard-boiled egg with black pepper.', 'dinner': 'Pan-seared cod with sautéed spinach and Brussels sprouts.', 'fluids': '2.5L water, 1 cup spearmint tea.'},
            {'day': 'Saturday', 'breakfast': 'Omelet with goat cheese, spinach, and tomatoes.', 'lunch': 'Grilled chicken salad with walnuts, avocado, and balsamic vinaigrette.', 'snack': 'A handful of almonds.', 'dinner': 'Grilled pork chop with roasted cauliflower and a side salad.', 'fluids': '2.5L water, 1 cup chamomile tea.'},
            {'day': 'Sunday', 'breakfast': 'Avocado toast on 1 slice of low-carb high-protein bread, topped with a fried egg.', 'lunch': 'Lemon herb baked chicken with a Mediterranean cucumber-tomato salad.', 'snack': 'Chia seed pudding.', 'dinner': 'Beef stir-fry with bell peppers, mushrooms, and broccoli (no rice).', 'fluids': '2.5L water, 1 cup spearmint tea.'}
        ]
    },
    'B': {
        'title': 'Anti-Inflammatory Insulin-Sensitizing Plan',
        'subtitle': 'Tailored for Lean PCOS Management (BMI < 25)',
        'description': 'Focuses on hormonal balance, reducing systemic inflammation, and optimizing insulin reception without caloric deficit.',
        'days': [
            {'day': 'Monday', 'breakfast': 'Oats cooked in water, topped with ground flaxseed, walnuts, and blueberries.', 'lunch': 'Quinoa bowl with chickpeas, roasted sweet potato, avocado, and tahini.', 'snack': 'Apple slices with pumpkin seed butter.', 'dinner': 'Baked salmon with roasted asparagus and quinoa.', 'fluids': '2.5L water, 1 cup spearmint tea.'},
            {'day': 'Tuesday', 'breakfast': 'Smoothie with spinach, half a banana, protein powder, and hemp seeds.', 'lunch': 'Lentil soup with a large side salad of mixed greens and olive oil.', 'snack': 'Greek yogurt with raw honey and pumpkin seeds.', 'dinner': 'Grilled chicken breast with sautéed kale and sweet potato mash.', 'fluids': '2.5L water, 1 cup green tea.'},
            {'day': 'Wednesday', 'breakfast': 'Avocado and poached eggs on sourdough bread.', 'lunch': 'Tuna and white bean salad over arugula with lemon vinaigrette.', 'snack': 'Handful of mixed walnuts and macadamia nuts.', 'dinner': 'Tempeh stir-fry with mixed vegetables and wild brown rice.', 'fluids': '2.5L water, 1 cup spearmint tea.'},
            {'day': 'Thursday', 'breakfast': 'Chia seed pudding with coconut milk, almonds, and strawberries.', 'lunch': 'Grilled chicken wrap with whole-wheat tortilla, hummus, and cucumber.', 'snack': 'Carrot sticks with guacamole.', 'dinner': 'Baked cod with roasted root vegetables (carrots, parsnips, beets).', 'fluids': '2.5L water, 1 cup turmeric-ginger tea.'},
            {'day': 'Friday', 'breakfast': 'Spinach, tomato, and feta omelet with 1 slice of sprouted grain toast.', 'lunch': 'Black bean and corn salad with grilled shrimp and cilantro-lime dressing.', 'snack': 'A cup of bone broth and a handful of pumpkin seeds.', 'dinner': 'Turkey stir-fry with green beans, bell peppers, and brown rice.', 'fluids': '2.5L water, 1 cup spearmint tea.'},
            {'day': 'Saturday', 'breakfast': 'Buckwheat pancakes topped with fresh berries and a drizzle of almond butter.', 'lunch': 'Salad with grilled salmon, spinach, walnuts, and strawberries.', 'snack': 'Greek yogurt with cinnamon.', 'dinner': 'Roasted chicken thighs with sweet potato wedges and steamed broccoli.', 'fluids': '2.5L water, 1 cup green tea.'},
            {'day': 'Sunday', 'breakfast': '2 scrambled eggs, sautéed mushrooms, spinach, and roasted cherry tomatoes.', 'lunch': 'Quinoa and black bean stuffed bell peppers topped with avocado slice.', 'snack': 'A handful of almonds and a plum.', 'dinner': 'Baked trout with lemon-herb seasoning, roasted asparagus, and wild rice.', 'fluids': '2.5L water, 1 cup spearmint tea.'}
        ]
    },
    'C': {
        'title': 'Calorie-Controlled Cardiovascular Health Plan',
        'subtitle': 'For General Weight Management & Heart Health (BMI ≥ 25, PCOS Negative)',
        'description': 'Focuses on gradual fat loss, lowering cholesterol, and supporting cardiovascular fitness.',
        'days': [
            {'day': 'Monday', 'breakfast': 'Steel-cut oats with almond milk, topped with sliced strawberries and chia seeds.', 'lunch': 'Whole-wheat pita stuffed with grilled chicken, hummus, and cucumber.', 'snack': '1 medium pear with a few almonds.', 'dinner': 'Baked turkey breast with a side of roasted sweet potato and steamed green beans.', 'fluids': '2.0L water, 1 cup green tea.'},
            {'day': 'Tuesday', 'breakfast': 'Egg white omelet with spinach and tomatoes, 1 slice of whole-wheat toast.', 'lunch': 'Mixed bean salad with olive oil, lemon juice, and chopped fresh herbs.', 'snack': 'Low-fat cottage cheese with pineapple chunks.', 'dinner': 'Grilled salmon with roasted asparagus and a small side of quinoa.', 'fluids': '2.0L water, 1 cup hibiscus tea.'},
            {'day': 'Wednesday', 'breakfast': 'Greek yogurt with blueberries, low-sugar granola, and ground flaxseed.', 'lunch': 'Tuna salad sandwich on whole-grain bread with lettuce and tomato.', 'snack': 'Celery sticks with peanut butter.', 'dinner': 'Chicken breast stir-fry with broccoli, carrots, and water chestnuts.', 'fluids': '2.0L water, 1 cup green tea.'},
            {'day': 'Thursday', 'breakfast': 'Smoothie with skim milk, strawberries, spinach, and a scoop of protein powder.', 'lunch': 'Quinoa bowl with black beans, corn, grilled zucchini, and salsa.', 'snack': '1 hard-boiled egg.', 'dinner': 'Baked cod with a side of brown rice and steamed broccoli.', 'fluids': '2.0L water, 1 cup chamomile tea.'},
            {'day': 'Friday', 'breakfast': 'Bran flakes with low-fat milk and sliced banana.', 'lunch': 'Turkey breast salad with mixed greens, dried cranberries, and light vinaigrette.', 'snack': 'Carrot sticks with hummus.', 'dinner': 'Grilled lean sirloin steak with a side of roasted cauliflower and green salad.', 'fluids': '2.0L water, 1 cup green tea.'},
            {'day': 'Saturday', 'breakfast': 'Whole-wheat English muffin with almond butter and banana slices.', 'lunch': 'Lentil salad with chopped bell peppers, cucumbers, and feta cheese.', 'snack': 'A handful of roasted pumpkin seeds.', 'dinner': 'Roasted chicken breast with Brussels sprouts and a small baked potato.', 'fluids': '2.0L water, 1 cup lemon tea.'},
            {'day': 'Sunday', 'breakfast': '2 poached eggs over sautéed spinach and mushrooms.', 'lunch': 'Grilled chicken wrap with mixed greens and low-fat tzatziki.', 'snack': 'Greek yogurt with cinnamon.', 'dinner': 'Baked shrimp with garlic, olive oil, roasted zucchini, and cherry tomatoes.', 'fluids': '2.0L water, 1 cup green tea.'}
        ]
    },
    'D': {
        'title': 'Balanced Preventative Nutrition Plan',
        'subtitle': 'For Active Wellness & General Hormone Support (BMI < 25, PCOS Negative)',
        'description': 'Maintains high energy levels, supports metabolic longevity, and provides comprehensive micronutrient coverage.',
        'days': [
            {'day': 'Monday', 'breakfast': '2 whole eggs scrambled with spinach, 1 slice of whole-grain toast with jam.', 'lunch': 'Quinoa bowl with mixed greens, avocado, cherry tomatoes, and grilled chicken.', 'snack': 'Greek yogurt with honey and sliced almonds.', 'dinner': 'Baked salmon with sweet potato mash and roasted asparagus.', 'fluids': '2.0L water, 1 cup herbal tea.'},
            {'day': 'Tuesday', 'breakfast': 'Oatmeal made with milk, topped with banana slices, walnuts, and honey.', 'lunch': 'Turkey breast sandwich on sprouted grain bread with avocado and microgreens.', 'snack': 'An orange and a handful of cashews.', 'dinner': 'Chicken breast cacciatore with tomatoes, bell peppers, and brown rice.', 'fluids': '2.0L water, 1 cup white tea.'},
            {'day': 'Wednesday', 'breakfast': 'Smoothie with almond milk, banana, spinach, peanut butter, and oats.', 'lunch': 'Lentil soup served with a slice of rustic whole-wheat sourdough.', 'snack': 'Apple slices with cheese slices.', 'dinner': 'Baked pork loin with garlic roasted potatoes and steamed green beans.', 'fluids': '2.0L water, 1 cup green tea.'},
            {'day': 'Thursday', 'breakfast': 'Breakfast burrito with scrambled eggs, black beans, and salsa in a whole-wheat wrap.', 'lunch': 'Salad with grilled shrimp, arugula, quinoa, orange segments, and vinaigrette.', 'snack': 'Chia seed pudding with coconut flakes.', 'dinner': 'Beef and vegetable stir-fry with snap peas, broccoli, and jasmine rice.', 'fluids': '2.0L water, 1 cup ginger tea.'},
            {'day': 'Friday', 'breakfast': 'French toast made with whole-wheat bread, topped with fresh berries and maple syrup.', 'lunch': 'Stuffed pita with chickpea salad, cucumber, tahini, and spinach.', 'snack': 'Hummus with pita chips and bell pepper slices.', 'dinner': 'Baked cod with a crust of herbs and breadcrumbs, side of roasted broccoli.', 'fluids': '2.0L water, 1 cup herbal tea.'},
            {'day': 'Saturday', 'breakfast': 'Omelet with cheddar cheese, ham, onions, and bell peppers.', 'lunch': 'Grilled chicken Caesar salad with whole-grain croutons.', 'snack': 'A handful of mixed berries and walnuts.', 'dinner': 'Grilled steak with roasted baby potatoes and asparagus.', 'fluids': '2.0L water, 1 cup green tea.'},
            {'day': 'Sunday', 'breakfast': 'Pancakes made with oat flour, topped with sliced banana and walnut butter.', 'lunch': 'Minestrone soup with a large side salad of mixed greens.', 'snack': 'Greek yogurt with berries.', 'dinner': 'Baked chicken breast with roasted Brussels sprouts and a side of wild rice.', 'fluids': '2.0L water, 1 cup chamomile tea.'}
        ]
    }
}

# --------------------------------------------------------------------------
# REPORTLAB PDF GENERATOR & CANVASES
# --------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_page_decorations(page_count)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Watermark
        self.setFont("Helvetica-Bold", 54)
        self.setFillColor(colors.HexColor('#FBEAF0'), 0.25)
        self.translate(A4[0] / 2.0, A4[1] / 2.0)
        self.rotate(45)
        self.drawCentredString(0, 0, "VERIFIED REPORT")
        
        self.restoreState()
        
        # Footer
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor('#6B7280'))
        self.drawString(36, 20, "PCOS Predict AI Diagnostics Group")
        self.drawRightString(A4[0] - 36, 20, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def generate_qr_code_flow(verification_url):
    qr = qrcode.QRCode(version=1, box_size=10, border=1)
    qr.add_data(verification_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def generate_pdf_certificate_file(assessment):
    pdf_path = os.path.join(app.root_path, 'uploads', 'certificates', f"{assessment.uuid}.pdf")
    
    try:
        host_url = request.host_url
    except Exception:
        host_url = "http://127.0.0.1:5000/"
        
    verify_url = f"{host_url}verify/{assessment.uuid}"
    qr_buf = generate_qr_code_flow(verify_url)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CertTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#FFFFFF'),
        alignment=TA_CENTER,
        spaceAfter=0
    )
    
    subtitle_style = ParagraphStyle(
        'CertSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#FBEAF0'),
        alignment=TA_CENTER,
        spaceAfter=0
    )
    
    patient_label_style = ParagraphStyle(
        'PatientLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        textColor=colors.HexColor('#555555'),
        alignment=TA_CENTER,
        spaceAfter=10
    )
    
    patient_name_style = ParagraphStyle(
        'PatientName',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#D4537E'),
        alignment=TA_CENTER,
        spaceAfter=8
    )
    
    body_text_style = ParagraphStyle(
        'CertBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#333333'),
        alignment=TA_CENTER
    )
    
    bold_label_style = ParagraphStyle(
        'BoldLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.HexColor('#444444')
    )
    
    val_text_style = ParagraphStyle(
        'ValueText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#333333')
    )
    
    outcome_style = ParagraphStyle(
        'OutcomeText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#E24B4A') if assessment.pcos_detected else colors.HexColor('#1D9E75')
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#888888'),
        alignment=TA_CENTER
    )
    
    story = []
    
    # 1. Title Banner
    header_data = [
        [Paragraph("PCOS Health Assessment Certificate", title_style)],
        [Spacer(1, 4)],
        [Paragraph("Polycystic Ovary Syndrome Clinical Risk Predictor", subtitle_style)]
    ]
    header_table = Table(header_data, colWidths=[523.27])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#D4537E')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 16),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 20))
    
    # 2. Certificate Frame
    story.append(Paragraph("This document certifies that", patient_label_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<u>{assessment.patient.name}</u>", patient_name_style))
    story.append(Spacer(1, 8))
    
    date_str = assessment.date.strftime('%B %d, %Y')
    story.append(Paragraph(f"has undergone a clinical AI-assisted risk screening on <b>{date_str}</b>.", body_text_style))
    story.append(Spacer(1, 20))
    
    # 3. Two-Column Results Box
    outcome_label = "PCOS Detected (High Clinical Correlates)" if assessment.pcos_detected else "No PCOS Detected (Low Clinical Correlates)"
    
    results_data = [
        [Paragraph("Date of Screening:", bold_label_style), Paragraph(date_str, val_text_style)],
        [Paragraph("Assessment Status:", bold_label_style), Paragraph(outcome_label, outcome_style)],
        [Paragraph("Risk Score (Probability):", bold_label_style), Paragraph(f"{assessment.risk_pct}%", val_text_style)],
        [Paragraph("Risk Category:", bold_label_style), Paragraph(f"<b>{assessment.risk_level} Risk</b>", val_text_style)],
        [Paragraph("Patient Age / BMI:", bold_label_style), Paragraph(f"{assessment.patient.age} yrs / {assessment.patient.bmi}", val_text_style)]
    ]
    
    left_table = Table(results_data, colWidths=[150, 240])
    left_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    qr_image = Image(qr_buf, width=90, height=90)
    
    right_data = [
        [qr_image],
        [Paragraph("Scan to Verify Report", ParagraphStyle('QRLabel', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor('#6B7280')))]
    ]
    right_table = Table(right_data, colWidths=[110])
    right_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    
    summary_box_data = [
        [left_table, right_table]
    ]
    summary_box_table = Table(summary_box_data, colWidths=[400, 123.27])
    summary_box_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FBEAF0')),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E7B5C7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_box_table)
    story.append(Spacer(1, 15))
    
    # 4. Parameters Grid
    details_title_style = ParagraphStyle(
        'DetailsTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor('#D4537E'),
        spaceAfter=6
    )
    story.append(Paragraph("Detailed Clinical Parameters Summary", details_title_style))
    
    param_data = [
        [Paragraph("<b>Parameter</b>", bold_label_style), Paragraph("<b>Value</b>", bold_label_style), Paragraph("<b>Reference Range</b>", bold_label_style),
         Paragraph("<b>Parameter</b>", bold_label_style), Paragraph("<b>Value</b>", bold_label_style), Paragraph("<b>Reference Range</b>", bold_label_style)],
        [Paragraph("LH Level", val_text_style), Paragraph(f"{assessment.lh} mIU/mL", val_text_style), Paragraph("1.0 - 12.0", val_text_style),
         Paragraph("FSH Level", val_text_style), Paragraph(f"{assessment.fsh} mIU/mL", val_text_style), Paragraph("1.0 - 9.0", val_text_style)],
        [Paragraph("AMH", val_text_style), Paragraph(f"{assessment.amh} ng/mL", val_text_style), Paragraph("1.5 - 4.5", val_text_style),
         Paragraph("Testosterone", val_text_style), Paragraph(f"{assessment.testosterone} ng/dL", val_text_style), Paragraph("15 - 70", val_text_style)],
        [Paragraph("Left Follicles", val_text_style), Paragraph(str(assessment.follicle_l), val_text_style), Paragraph("< 10", val_text_style),
         Paragraph("Right Follicles", val_text_style), Paragraph(str(assessment.follicle_r), val_text_style), Paragraph("< 10", val_text_style)],
        [Paragraph("Cycle Length", val_text_style), Paragraph(f"{assessment.cycle_length} days", val_text_style), Paragraph("21 - 35 days", val_text_style),
         Paragraph("Symptom Score", val_text_style), Paragraph(f"{round(assessment.symptom_score, 2)}", val_text_style), Paragraph("0.2 - 0.5 (Low)", val_text_style)]
    ]
    param_table = Table(param_data, colWidths=[100, 75, 85, 100, 75, 88.27])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 15))
    
    # 5. Clinical Summaries
    ai_summary_title = ParagraphStyle(
        'AISummaryTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor('#7C3AED'),
        spaceAfter=4
    )
    ai_body_style = ParagraphStyle(
        'AIBody',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4B5563')
    )
    
    ai_notes_data = [
        [Paragraph("AI Clinical Synthesis & Lifestyle Note", ai_summary_title)],
        [Paragraph(f"<b>Assessment Summary:</b> {assessment.gemini_summary or assessment.clinical_summary}", ai_body_style)],
        [Spacer(1, 4)],
        [Paragraph(f"<b>Lifestyle Recommendations:</b> {assessment.doctor_note or 'No custom recommendations available.'}", ai_body_style)]
    ]
    ai_notes_table = Table(ai_notes_data, colWidths=[523.27])
    ai_notes_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F3FF')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDD6FE')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(ai_notes_table)
    story.append(Spacer(1, 15))
    
    # 6. Signature blocks
    sig_name_style = ParagraphStyle(
        'SigName',
        parent=styles['Normal'],
        fontName='Courier-BoldOblique',
        fontSize=11,
        textColor=colors.HexColor('#B23D61'),
        alignment=TA_CENTER
    )
    sig_label_style = ParagraphStyle(
        'SigLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#6B7280'),
        alignment=TA_CENTER
    )
    
    sig_data = [
        [
            Paragraph("<i>PCOS Predict Engine</i>", sig_name_style),
            Paragraph("<i>Dr. Sophia Vance</i>", sig_name_style)
        ],
        [
            Paragraph("_____________________________<br/>AI Verification Signature", sig_label_style),
            Paragraph("_____________________________<br/>Medical Director, PCOS Group", sig_label_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[261.6, 261.6])
    sig_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 20))
    
    # 7. Legal Disclaimer
    disclaimer_text = (
        "<b>Clinical Disclaimer:</b> This certificate represents a statistical machine learning analysis of patient symptoms "
        "and hormone values. It does not replace a physical pelvic ultrasound or official diagnosis by a qualified obstetrician-gynecologist. "
        "All recommendations should be discussed with a licensed healthcare provider before implementation."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)

# --------------------------------------------------------------------------
# SYSTEM ROUTES & VIEWS
# --------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    if not model_loaded:
        load_ml_components()
    return render_template('index.html', model_loaded=model_loaded)

# Registration View
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template('auth/register.html')
            
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template('auth/register.html')
            
        # Verify existing user
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return render_template('auth/register.html')
            
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return render_template('auth/register.html')
            
        hashed_pwd = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_pwd)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error saving user: {e}", "error")
            
    return render_template('auth/register.html')

# Login View
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "error")
            
    return render_template('auth/login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

# User Dashboard View
@app.route('/dashboard')
@login_required
def dashboard():
    # Summarize assessment counts, risk categories, history list
    assessments = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.date.desc()).all()
    patients_count = Patient.query.filter_by(user_id=current_user.id).count()
    assessments_count = len(assessments)
    
    # Calculate stats
    high_count = sum(1 for a in assessments if a.risk_level == "High")
    mod_count = sum(1 for a in assessments if a.risk_level == "Moderate")
    low_count = sum(1 for a in assessments if a.risk_level == "Low")
    
    recent_assessments = assessments[:5]
    
    return render_template(
        'dashboard.html',
        patients_count=patients_count,
        assessments_count=assessments_count,
        high_count=high_count,
        mod_count=mod_count,
        low_count=low_count,
        recent_assessments=recent_assessments
    )

# Longitudinal Assessment History View
@app.route('/history')
@login_required
def history():
    assessments = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.date.asc()).all()
    
    # Prepare historical line chart data for Chart.js
    dates = [a.date.strftime('%Y-%m-%d') for a in assessments]
    scores = [a.risk_pct for a in assessments]
    names = [a.patient.name for a in assessments]
    
    chart_data = {
        'labels': dates,
        'scores': scores,
        'names': names
    }
    
    return render_template('history.html', assessments=reversed(assessments), chart_data=json.dumps(chart_data))

# Predict Action Route
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    global model, scaler, model_loaded
    if not model_loaded:
        load_ml_components()
        if not model_loaded:
            flash("Machine learning model files are not trained or loaded. Contact system admin.", "error")
            return redirect(url_for('index'))

    try:
        # Collect basic info
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        age_str = request.form.get('age', '')
        height_str = request.form.get('height', '')
        weight_str = request.form.get('weight', '')
        
        # Collect cycle & hormonal details
        cycle_reg = request.form.get('cycle_reg', 'regular')
        cycle_length_str = request.form.get('cycle_length', '28')
        fsh_str = request.form.get('fsh', '')
        lh_str = request.form.get('lh', '')
        amh_str = request.form.get('amh', '')
        testosterone_str = request.form.get('testosterone', '')
        follicle_l_str = request.form.get('follicle_l', '')
        follicle_r_str = request.form.get('follicle_r', '')
        
        # Collect checkboxes/symptoms (from 1 to 5 sliders)
        acne = int(request.form.get('acne', 1))
        hair_loss = int(request.form.get('hair_loss', 1))
        hirsutism = int(request.form.get('hirsutism', 1))
        weight_gain = int(request.form.get('weight_gain', 1))
        darkening = int(request.form.get('darkening', 1))
        cycle = int(request.form.get('cycle', 1))
        fatigue = int(request.form.get('fatigue', 1))
        mood_swings = int(request.form.get('mood_swings', 1))
        headaches = int(request.form.get('headaches', 1))
        conceiving = int(request.form.get('conceiving', 1))
        
        fast_food = 1 if request.form.get('fast_food') == '1' else 0

        # Validate inputs
        if not name:
            flash("Please provide the patient's full name.", "error")
            return redirect(url_for('index'))
            
        try:
            age = float(age_str)
            height = float(height_str)
            weight = float(weight_str)
            cycle_length = float(cycle_length_str)
            fsh = float(fsh_str)
            lh = float(lh_str)
            amh = float(amh_str)
            testosterone = float(testosterone_str)
            follicle_l = int(follicle_l_str)
            follicle_r = int(follicle_r_str)
        except ValueError:
            flash("Invalid numeric format in clinical features.", "error")
            return redirect(url_for('index'))
            
        # Physical validations
        if age < 13 or age > 55 or height < 100 or height > 220 or weight < 30 or weight > 200:
            flash("Input validation failed: Check age (13-55), height (100-220), and weight (30-200).", "error")
            return redirect(url_for('index'))

        # Auto-calculate BMI & LH/FSH ratio
        height_m = height / 100.0
        bmi = round(weight / (height_m ** 2), 2)
        lh_fsh_ratio = round(lh / fsh, 3) if fsh > 0 else 0.0
        
        # Calculate standardized symptom_score
        symptom_score = (acne + hair_loss + hirsutism + weight_gain + darkening + cycle + fatigue + mood_swings + headaches + conceiving) / 50.0
        
        # Map sliders (>= 3 is positive, else negative) to binary representations for the ensemble model
        bin_darkening = 1 if darkening >= 3 else 0
        bin_hair_growth = 1 if hirsutism >= 3 else 0
        bin_weight_gain = 1 if weight_gain >= 3 else 0
        bin_cycle_reg = 1 if cycle >= 3 else 0
        
        # Build features list in the exact order model was trained on
        feature_names = [
            'Age', 'BMI', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'AMH(ng/mL)',
            'Testosterone(ng/dL)', 'Follicle No.(L)', 'Follicle No.(R)',
            'Skin_darkening', 'hair_growth', 'Weight_gain', 'Cycle_length(days)', 'Fast_food', 'symptom_score'
        ]
        
        features_list = [
            age, bmi, bin_cycle_reg, fsh, lh, amh, testosterone,
            follicle_l, follicle_r, bin_darkening, bin_hair_growth,
            bin_weight_gain, cycle_length, fast_food, symptom_score
        ]
        
        features_df = pd.DataFrame([features_list], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        
        # Model Inference
        pcos_detected_class = int(model.predict(features_scaled)[0])
        risk_prob = float(model.predict_proba(features_scaled)[0][1])
        risk_pct = round(risk_prob * 100, 1)
        
        if risk_pct < 35:
            risk_level = "Low"
        elif risk_pct <= 65:
            risk_level = "Moderate"
        else:
            risk_level = "High"
            
        pcos_detected = True if pcos_detected_class == 1 else False
        
        # 1. SHAP Interpretations
        rf_model = model.named_estimators_['rf']
        up_factors, down_factors = get_shap_explanation(rf_model, scaler, features_df, feature_names)
        
        # 2. Call Google Gemini Report Writer
        patient_payload = {
            'name': name, 'age': age, 'bmi': bmi, 'pcos_detected': pcos_detected, 'risk_pct': risk_pct,
            'follicle_l': follicle_l, 'follicle_r': follicle_r, 'amh': amh, 'testosterone': testosterone, 'lh_fsh_ratio': lh_fsh_ratio,
            'acne': acne, 'hair_loss': hair_loss, 'hirsutism': hirsutism, 'weight_gain': weight_gain, 'darkening': darkening, 'cycle': cycle
        }
        gemini_result = call_gemini_api(app.config['GEMINI_API_KEY'], patient_payload)
        
        # Standard fallback clinical description
        if pcos_detected:
            clinical_summary = (
                f"Elevated risk signals detected. Left ovary follicle count is {follicle_l} and right is {follicle_r}. "
                f"Luteinizing Hormone (LH) is elevated relative to FSH. Physical symptoms including hirsutism (severity: {hirsutism}) "
                f"and cycle irregularities (severity: {cycle}) align with Polycystic Ovary Syndrome characteristics."
            )
        else:
            clinical_summary = (
                f"Patient exhibits standard baseline levels. LH/FSH ratio is normal ({lh_fsh_ratio}). Ovarian follicles are "
                f"within range ({follicle_l + follicle_r} total). Standard preventative metabolic protocols are suggested."
            )

        # 3. Save Patient and Assessment Records
        # Query if patient name already exists for current user
        patient = Patient.query.filter_by(user_id=current_user.id, name=name).first()
        if not patient:
            patient = Patient(user_id=current_user.id, name=name, email=email if email else None, age=age, height=height, weight=weight, bmi=bmi)
            db.session.add(patient)
            db.session.flush() # Populate patient ID
        else:
            # Update physical records
            patient.age = age
            patient.height = height
            patient.weight = weight
            patient.bmi = bmi
            if email:
                patient.email = email
            
        assessment = Assessment(
            user_id=current_user.id,
            patient_id=patient.id,
            cycle_length=cycle_length,
            fsh=fsh,
            lh=lh,
            amh=amh,
            testosterone=testosterone,
            follicle_l=follicle_l,
            follicle_r=follicle_r,
            fast_food=fast_food,
            acne=acne,
            hair_loss=hair_loss,
            hirsutism=hirsutism,
            weight_gain=weight_gain,
            darkening=darkening,
            cycle=cycle,
            fatigue=fatigue,
            mood_swings=mood_swings,
            headaches=headaches,
            conceiving=conceiving,
            symptom_score=symptom_score,
            risk_pct=risk_pct,
            risk_level=risk_level,
            pcos_detected=pcos_detected,
            clinical_summary=clinical_summary,
            gemini_summary=gemini_result.get('summary'),
            doctor_note=gemini_result.get('doctor_note'),
            tags=",".join(gemini_result.get('tags', []))
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        # 4. Generate & Archive the PDF Certificate
        generate_pdf_certificate_file(assessment)
        
        # Clinical parameters progress indicators
        indicators = {
            'lh_fsh': {
                'value': lh_fsh_ratio,
                'min': 0.1, 'max': 3.5,
                'pct': min(100, max(0, int((lh_fsh_ratio / 3.0) * 100))),
                'status': 'normal' if lh_fsh_ratio < 1.5 else ('borderline' if lh_fsh_ratio < 2.0 else 'elevated')
            },
            'amh': {
                'value': amh,
                'min': 0.5, 'max': 15.0,
                'pct': min(100, max(0, int((amh / 12.0) * 100))),
                'status': 'normal' if amh < 4.0 else ('borderline' if amh <= 6.0 else 'elevated')
            },
            'testosterone': {
                'value': testosterone,
                'min': 10, 'max': 150,
                'pct': min(100, max(0, int((testosterone / 120.0) * 100))),
                'status': 'normal' if testosterone < 55 else ('borderline' if testosterone <= 70 else 'elevated')
            },
            'bmi': {
                'value': bmi,
                'min': 15, 'max': 40,
                'pct': min(100, max(0, int(((bmi - 15) / 25.0) * 100))),
                'status': 'normal' if bmi < 25 else ('borderline' if bmi < 30 else 'elevated')
            },
            'follicles': {
                'value': follicle_l + follicle_r,
                'min': 0, 'max': 30,
                'pct': min(100, max(0, int(((follicle_l + follicle_r) / 24.0) * 100))),
                'status': 'normal' if (follicle_l + follicle_r) < 12 else ('borderline' if (follicle_l + follicle_r) <= 16 else 'elevated')
            },
            'symptoms': {
                'value': round(symptom_score * 50, 0),
                'pct': min(100, max(0, int(((symptom_score - 0.2) / 0.8) * 100))),
                'status': 'normal' if symptom_score < 0.4 else ('borderline' if symptom_score <= 0.6 else 'elevated')
            }
        }
        
        result_data = {
            'assessment_id': assessment.id,
            'name': name,
            'age': age,
            'weight': weight,
            'height': height,
            'bmi': bmi,
            'lh_fsh_ratio': lh_fsh_ratio,
            'risk_pct': risk_pct,
            'risk_level': risk_level,
            'pcos_detected': pcos_detected,
            'clinical_summary': gemini_result.get('summary') or clinical_summary,
            'doctor_note': gemini_result.get('doctor_note'),
            'tags': gemini_result.get('tags', []),
            'indicators': indicators,
            'up_factors': up_factors,
            'down_factors': down_factors
        }
        
        # Store metadata in session for diet recommendations
        session['last_result'] = {
            'assessment_id': assessment.id,
            'name': name,
            'bmi': bmi,
            'pcos_detected': pcos_detected
        }
        
        return render_template('result.html', result=result_data)
        
    except Exception as e:
        db.session.rollback()
        print(f"Prediction error: {e}")
        flash(f"Error during diagnostic execution: {e}", "error")
        return redirect(url_for('index'))

# General Wellness recommendations
@app.route('/wellness')
@login_required
def wellness():
    detected_param = request.args.get('detected')
    if detected_param is not None:
        pcos_detected = detected_param.lower() in ['true', '1']
    else:
        last_res = session.get('last_result')
        pcos_detected = last_res.get('pcos_detected', False) if last_res else False
        
    last_res = session.get('last_result')
    name = last_res.get('name', 'Patient') if last_res else 'Patient'
    assessment_id = last_res.get('assessment_id') if last_res else None
    
    if not assessment_id:
        latest_assessment = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.date.desc()).first()
        if latest_assessment:
            assessment_id = latest_assessment.id
            pcos_detected = latest_assessment.pcos_detected
            name = latest_assessment.patient.name
            
    return render_template(
        'wellness.html',
        pcos_detected=pcos_detected,
        name=name,
        assessment_id=assessment_id
    )

# Personalized Nutritional Guidance (Rotational Meal Plans)
@app.route('/meal-plan')
@login_required
def meal_plan():
    last_res = session.get('last_result')
    if not last_res:
        # Load from DB history if available
        latest_assessment = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.date.desc()).first()
        if latest_assessment:
            last_res = {
                'name': latest_assessment.patient.name,
                'bmi': latest_assessment.patient.bmi,
                'pcos_detected': latest_assessment.pcos_detected
            }
        else:
            flash("Take a PCOS assessment first to unlock your personalized meal plan.", "error")
            return redirect(url_for('index'))
            
    pcos = last_res['pcos_detected']
    bmi = last_res['bmi']
    
    if pcos and bmi >= 25.0:
        plan_key = 'A'
    elif pcos and bmi < 25.0:
        plan_key = 'B'
    elif not pcos and bmi >= 25.0:
        plan_key = 'C'
    else:
        plan_key = 'D'
        
    plan = MEAL_PLANS[plan_key]
    
    return render_template('meal_plan.html', plan=plan, patient_name=last_res['name'])

# Certificate Download Route
@app.route('/download-certificate/<int:assessment_id>')
@login_required
def download_certificate(assessment_id):
    assessment = Assessment.query.get_or_404(assessment_id)
    if assessment.user_id != current_user.id:
        flash("Unauthorized access to report.", "error")
        return redirect(url_for('dashboard'))
        
    pdf_path = os.path.join(app.root_path, 'uploads', 'certificates', f"{assessment.uuid}.pdf")
    
    # Re-generate if it doesn't exist
    if not os.path.exists(pdf_path):
        generate_pdf_certificate_file(assessment)
        
    safe_name = assessment.patient.name.replace(" ", "_")
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"PCOS_Certificate_{safe_name}.pdf",
        mimetype='application/pdf'
    )

# Email Certificate SMTP dispatch
@app.route('/email-certificate', methods=['POST'])
@login_required
def email_certificate():
    assessment_id = request.form.get('assessment_id')
    recipient_email = request.form.get('email', '').strip()
    
    if not recipient_email or not assessment_id:
        return jsonify({'success': False, 'message': 'Email address and Assessment ID are required.'}), 400
        
    assessment = Assessment.query.get_or_404(int(assessment_id))
    if assessment.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Unauthorized access to report.'}), 403
        
    pdf_path = os.path.join(app.root_path, 'uploads', 'certificates', f"{assessment.uuid}.pdf")
    if not os.path.exists(pdf_path):
        generate_pdf_certificate_file(assessment)
        
    try:
        msg = Message(
            subject=f"PCOS Predict - Health Screening Report: {assessment.patient.name}",
            recipients=[recipient_email],
            body=(
                f"Dear {assessment.patient.name},\n\n"
                f"Your clinical AI-assisted PCOS risk screening report has been compiled.\n\n"
                f"Screening Summary:\n"
                f" - Date: {assessment.date.strftime('%B %d, %Y')}\n"
                f" - Risk Probability: {assessment.risk_pct}%\n"
                f" - Classification: {assessment.risk_level} Risk ({'PCOS Detected' if assessment.pcos_detected else 'No PCOS Detected'})\n\n"
                f"Please review the attached official PDF document for full parameters, risk attributions, and dietary plans.\n"
                f"Verify your report online at: {url_for('verify_report', uuid=assessment.uuid, _external=True)}\n\n"
                f"Wishing you the best of health,\n"
                f"The PCOS Predict Group"
            )
        )
        with open(pdf_path, 'rb') as fp:
            msg.attach(
                filename=f"PCOS_Report_{assessment.patient.name.replace(' ', '_')}.pdf",
                content_type="application/pdf",
                data=fp.read()
            )
        mail.send(msg)
        return jsonify({'success': True, 'message': f'Report email successfully sent to {recipient_email}!'})
    except Exception as e:
        print(f"SMTP failed: {e}")
        return jsonify({'success': False, 'message': f'Failed to send email. Check SMTP settings. Error: {str(e)}'}), 500

# Public Report Verification URL
@app.route('/verify/<string:uuid>')
def verify_report(uuid):
    assessment = Assessment.query.filter_by(uuid=uuid).first_or_404()
    return render_template('verify.html', assessment=assessment)

# Plotly Statistics view
@app.route('/statistics')
@login_required
def statistics():
    csv_path = 'data/PCOS_data.csv'
    if not os.path.exists(csv_path):
        flash("Dataset file data/PCOS_data.csv not found. Stats dashboard cannot compile.", "error")
        return redirect(url_for('dashboard'))
        
    df = pd.read_csv(csv_path)
    # Strip headers
    df.columns = df.columns.str.strip()
    
    # Coerce clinical and symptomatic columns to numeric to handle strings like 'a'
    cols_to_coerce = [
        'PCOS (Y/N)', 'Age', 'Age (yrs)', 'BMI', 'LH(mIU/mL)', 'FSH(mIU/mL)',
        'AMH(ng/mL)', 'Testosterone(ng/dL)', 'Weight gain(Y/N)',
        'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)'
    ]
    for col in cols_to_coerce:
        for actual_col in df.columns:
            if col.strip().lower() == actual_col.strip().lower():
                df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
    
    # Check target
    target_col = 'PCOS (Y/N)'
    if target_col not in df.columns:
        # Fallback to standard columns rename
        df = df.rename(columns={'PCOS': 'PCOS (Y/N)'})
        
    df['PCOS (Y/N)'] = df['PCOS (Y/N)'].fillna(0).astype(int)
    
    # 1. PCOS Class split Donut Chart
    pcos_counts = df['PCOS (Y/N)'].value_counts().reset_index()
    pcos_counts.columns = ['Status', 'Count']
    pcos_counts['Status'] = pcos_counts['Status'].map({1: 'PCOS Positive', 0: 'Non-PCOS'})
    fig_donut = px.pie(pcos_counts, values='Count', names='Status', hole=0.5,
                       color_discrete_sequence=['#D4537E', '#1D9E75'],
                       title='Dataset Classification Split (PCOS vs Non-PCOS)')
    fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    donut_json = json.dumps(fig_donut, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Age distribution Histogram
    age_col = ' Age (yrs)' if ' Age (yrs)' in df.columns else ('Age' if 'Age' in df.columns else None)
    if age_col:
        fig_age = px.histogram(df, x=age_col, color='PCOS (Y/N)', nbins=15,
                               color_discrete_map={1: '#D4537E', 0: '#1D9E75'},
                               title='Age Distribution of Patients')
        fig_age.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        age_json = json.dumps(fig_age, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        age_json = '{}'

    # 3. LH vs FSH Scatter
    lh_col = 'LH(mIU/mL)'
    fsh_col = 'FSH(mIU/mL)'
    if lh_col in df.columns and fsh_col in df.columns:
        df_clean = df.dropna(subset=[lh_col, fsh_col])
        fig_scatter = px.scatter(df_clean, x=fsh_col, y=lh_col, color='PCOS (Y/N)',
                                 color_discrete_map={1: '#D4537E', 0: '#1D9E75'},
                                 title='LH vs FSH Hormonal Correlation')
        fig_scatter.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        scatter_json = json.dumps(fig_scatter, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        scatter_json = '{}'
        
    # 4. BMI Box Plot
    bmi_col = 'BMI'
    if bmi_col in df.columns:
        fig_bmi = px.box(df, y=bmi_col, x='PCOS (Y/N)', color='PCOS (Y/N)',
                         color_discrete_map={1: '#D4537E', 0: '#1D9E75'},
                         title='BMI Distribution boxplot')
        fig_bmi.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        bmi_json = json.dumps(fig_bmi, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        bmi_json = '{}'
        
    # 5. Correlation Heatmap
    corr_cols = [c for c in [' Age (yrs)', 'Age', 'BMI', 'LH(mIU/mL)', 'FSH(mIU/mL)', 'AMH(ng/mL)', 'Testosterone(ng/dL)'] if c in df.columns]
    if corr_cols:
        corr_matrix = df[corr_cols].corr()
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        fig_heatmap.update_layout(title='Endocrine Metrics Correlation Matrix', margin=dict(t=40, b=20, l=20, r=20))
        heatmap_json = json.dumps(fig_heatmap, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        heatmap_json = '{}'

    # 6. Symptom Frequency Bar Chart (among PCOS positive patients)
    symptom_cols = {
        'Weight gain(Y/N)': 'Weight Gain',
        'hair growth(Y/N)': 'Hirsutism',
        'Skin darkening (Y/N)': 'Skin Darkening',
        'Hair loss(Y/N)': 'Hair Loss',
        'Pimples(Y/N)': 'Acne'
    }
    present_symptoms = {k: v for k, v in symptom_cols.items() if k in df.columns}
    
    if present_symptoms:
        pcos_pos = df[df['PCOS (Y/N)'] == 1]
        freq_list = []
        for raw_col, clean_lbl in present_symptoms.items():
            rate = (pcos_pos[raw_col] == 1).mean() * 100
            freq_list.append({'Symptom': clean_lbl, 'Frequency (%)': round(rate, 1)})
            
        freq_df = pd.DataFrame(freq_list).sort_values(by='Frequency (%)', ascending=False)
        fig_symptoms = px.bar(freq_df, x='Symptom', y='Frequency (%)',
                              color_discrete_sequence=['#D4537E'],
                              title='Physical Symptom Frequency in PCOS Positive Cohort')
        fig_symptoms.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        symptoms_json = json.dumps(fig_symptoms, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        symptoms_json = '{}'

    return render_template(
        'statistics.html',
        donut_json=donut_json,
        age_json=age_json,
        scatter_json=scatter_json,
        bmi_json=bmi_json,
        heatmap_json=heatmap_json,
        symptoms_json=symptoms_json
    )

# --------------------------------------------------------------------------
# RATE-LIMITED PUBLIC REST API (v1)
# --------------------------------------------------------------------------
def check_api_key(api_key):
    return api_key in app.config['API_KEYS'].values()

@app.route('/api/v1/health', methods=['GET'])
@limiter.limit("100/hour")
def api_health():
    return jsonify({
        'status': 'healthy',
        'database_connected': True,
        'model_loaded': model_loaded,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/v1/features', methods=['GET'])
@limiter.limit("100/hour")
def api_features():
    return jsonify({
        'required_features': [
            {'name': 'Age', 'type': 'float', 'range': '13 - 55', 'description': 'Age of the patient'},
            {'name': 'BMI', 'type': 'float', 'range': '15 - 40', 'description': 'Body Mass Index'},
            {'name': 'Cycle(R/I)', 'type': 'int', 'range': '0 (Regular) or 1 (Irregular)', 'description': 'Menstrual cycle regularity'},
            {'name': 'FSH(mIU/mL)', 'type': 'float', 'range': '0.1 - 20.0', 'description': 'Follicle-Stimulating Hormone'},
            {'name': 'LH(mIU/mL)', 'type': 'float', 'range': '0.1 - 20.0', 'description': 'Luteinizing Hormone'},
            {'name': 'AMH(ng/mL)', 'type': 'float', 'range': '0.1 - 25.0', 'description': 'Anti-Müllerian Hormone'},
            {'name': 'Testosterone(ng/dL)', 'type': 'float', 'range': '10.0 - 180.0', 'description': 'Total Testosterone level'},
            {'name': 'Follicle No.(L)', 'type': 'int', 'range': '0 - 30', 'description': 'Follicle count in left ovary'},
            {'name': 'Follicle No.(R)', 'type': 'int', 'range': '0 - 30', 'description': 'Follicle count in right ovary'},
            {'name': 'Skin_darkening', 'type': 'int', 'range': '0 or 1', 'description': 'Presence of skin darkening / acanthosis nigricans'},
            {'name': 'hair_growth', 'type': 'int', 'range': '0 or 1', 'description': 'Presence of hirsutism'},
            {'name': 'Weight_gain', 'type': 'int', 'range': '0 or 1', 'description': 'Presence of weight gain'},
            {'name': 'Cycle_length(days)', 'type': 'float', 'range': '2 - 10', 'description': 'Average bleeding duration'},
            {'name': 'Fast_food', 'type': 'int', 'range': '0 or 1', 'description': 'Presence of regular fast food eating habits'},
            {'name': 'symptom_score', 'type': 'float', 'range': '0.2 - 1.0', 'description': 'Calculated cumulative severity index'}
        ]
    })

@app.route('/api/v1/predict', methods=['POST'])
@limiter.limit("100/hour")
def api_predict():
    start_time = time.time()
    
    # 1. API-Key Authentication
    api_key = request.headers.get('X-API-Key')
    if not api_key or not check_api_key(api_key):
        return jsonify({'error': 'Unauthorized: Invalid or missing X-API-Key header.'}), 401
        
    if not model_loaded:
        return jsonify({'error': 'Model ensemble is not loaded on the host server.'}), 503
        
    # 2. Parse payload
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request: Body must be valid JSON.'}), 400
        
    feature_names = [
        'Age', 'BMI', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'AMH(ng/mL)',
        'Testosterone(ng/dL)', 'Follicle No.(L)', 'Follicle No.(R)',
        'Skin_darkening', 'hair_growth', 'Weight_gain', 'Cycle_length(days)', 'Fast_food', 'symptom_score'
    ]
    
    # Extract features, checking that all required inputs are present
    missing_fields = [f for f in feature_names if f not in data]
    if missing_fields:
        return jsonify({'error': f'Missing required clinical features: {missing_fields}'}), 400
        
    try:
        features_list = [float(data[f]) for f in feature_names]
    except ValueError:
        return jsonify({'error': 'Invalid request: All clinical feature inputs must be numeric values.'}), 400
        
    try:
        # Scale & Inference
        features_df = pd.DataFrame([features_list], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        
        pcos_detected_class = int(model.predict(features_scaled)[0])
        risk_prob = float(model.predict_proba(features_scaled)[0][1])
        risk_pct = round(risk_prob * 100, 1)
        
        if risk_pct < 35:
            risk_level = "Low"
        elif risk_pct <= 65:
            risk_level = "Moderate"
        else:
            risk_level = "High"
            
        pcos_detected = True if pcos_detected_class == 1 else False
        
        # Format response
        response_payload = {
            'risk_percentage': risk_pct,
            'risk_level': risk_level,
            'pcos_detected': pcos_detected,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log transaction details
        exec_time = int((time.time() - start_time) * 1000)
        log_api_transaction(api_key, data, pcos_detected, risk_pct, exec_time)
        
        return jsonify(response_payload)
        
    except Exception as e:
        return jsonify({'error': f'API diagnostics execution failed: {str(e)}'}), 500

# Appointment Booking Route
@app.route('/appointment', methods=['GET', 'POST'])
@login_required
def appointment():
    if request.method == 'POST':
        doctor_name = request.form.get('doctor_name')
        appointment_date = request.form.get('appointment_date')
        appointment_time = request.form.get('appointment_time')
        patient_notes = request.form.get('notes', '')
        
        flash(f"Appointment request successfully sent to {doctor_name} for {appointment_date} at {appointment_time}! Our clinic will call you shortly to confirm.", "success")
        return redirect(url_for('dashboard'))
        
    doctors = [
        {'name': 'Dr. Priya Sharma', 'specialty': 'Gynecologist & Obstetrician', 'experience': '12 Years', 'location': 'Delhi Clinic'},
        {'name': 'Dr. Anjali Mehta', 'specialty': 'Reproductive Endocrinologist', 'experience': '15 Years', 'location': 'Mumbai Medical Center'},
        {'name': 'Dr. Sneha Reddy', 'specialty': 'Fertility Specialist', 'experience': '10 Years', 'location': 'Bangalore Health Hub'}
    ]
    return render_template('appointment.html', doctors=doctors)

# --------------------------------------------------------------------------
# DATABASE INITIALIZATION SEEDER
# --------------------------------------------------------------------------
with app.app_context():
    db.create_all()
    # Check default doctor account
    if not User.query.filter_by(username='doctor').first():
        hashed = generate_password_hash('doctor123')
        default_doctor = User(username='doctor', email='doctor@pcospredict.com', password_hash=hashed)
        db.session.add(default_doctor)
        db.session.commit()

if __name__ == '__main__':
    # Local host bind
    app.run(debug=True, port=5000)
