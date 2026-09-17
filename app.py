import os
from dotenv import load_dotenv
load_dotenv()
import io
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

import model
import db
import llm
import email_service

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static"
)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_florascann_key")

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return db.get_user_by_id(user_id)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB maximum upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# Safe upload directory handling for Vercel serverless read-only filesystem
import tempfile
IS_VERCEL = bool(os.environ.get("VERCEL"))
if IS_VERCEL:
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "uploads")
else:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception:
    pass
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Auth Routes ---
@app.route("/", methods=["GET"])
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = db.get_user_by_email(email)
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "error")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        
        if db.get_user_by_email(email):
            flash("Email already registered", "error")
        else:
            db.create_user(email, password, name)
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
            
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# --- Page Routes ---
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")

@app.route("/history_page")
@login_required
def history_page():
    return render_template("history.html")

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html")

@app.route("/profile/update", methods=["POST"])
@login_required
def update_profile():
    name = request.form.get("name", "").strip()
    notification_email = request.form.get("notification_email", "").strip()
    resend_api_key = request.form.get("resend_api_key", "").strip()
    
    db.update_user_profile(
        user_id=current_user.id,
        name=name if name else None,
        notification_email=notification_email if notification_email else None,
        resend_api_key=resend_api_key if resend_api_key else None
    )
    flash("Profile updated successfully!", "success")
    return redirect(url_for("profile"))


# --- API Routes ---
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        json_data = request.get_json(silent=True) or {}
        image = None
        send_email = False
        
        if "file" in request.files:
            if request.form:
                send_email = request.form.get("send_email", "false").lower() == "true"
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected"}), 400
            if not allowed_file(file.filename):
                return jsonify({"success": False, "error": "Invalid format"}), 400
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Check JSON
            send_email = json_data.get("send_email", False)
            if "image" in json_data:
                b64_data = json_data["image"]
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif "sample" in json_data:
                sample_name = secure_filename(json_data["sample"])
                sample_path = os.path.join(BASE_DIR, "static", "samples", sample_name)
                image = Image.open(sample_path)
            
        if not image:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Execute prediction
        results = model.predict_disease(image, top_k=5)
        
        # Save to DB
        pred_id = db.save_prediction(
            plant=results['plant'],
            disease=results['disease'],
            confidence=results['confidence'],
            user_id=current_user.id
        )
        results['prediction_id'] = pred_id
        
        # Optional: Generate recommendation and send email
        if send_email:
            rec = llm.get_recommendation(results['plant'], results['disease'])
            # Use profile settings: notification_email and resend_api_key
            to_email = current_user.notification_email or current_user.email
            user_api_key = current_user.resend_api_key
            success, msg, delivered_to = email_service.send_prediction_email(
                to_email, results['plant'], results['disease'], results['confidence'], rec, api_key=user_api_key
            )
            results['email_status'] = {
                "success": success,
                "message": msg,
                "delivered_to": delivered_to
            }

        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"Error: {str(e)}"}), 500

@app.route("/api/send-email-report", methods=["POST"])
@login_required
def send_email_report():
    try:
        data = request.get_json(silent=True) or {}
        pred_id = data.get("prediction_id")
        
        session_db = db.SessionLocal()
        try:
            if pred_id:
                pred = session_db.query(db.Prediction).filter_by(id=pred_id, user_id=current_user.id).first()
            else:
                pred = session_db.query(db.Prediction).filter_by(user_id=current_user.id).order_by(db.Prediction.timestamp.desc()).first()
                
            if not pred:
                return jsonify({"success": False, "error": "No prediction record found"}), 404
                
            plant = pred.plant_name
            disease = pred.disease_name
            confidence = pred.confidence
            pred_record_id = pred.id
        finally:
            session_db.close()
            
        rec = llm.get_recommendation(plant, disease)
        to_email = current_user.notification_email or current_user.email
        user_api_key = current_user.resend_api_key
        
        success, msg, delivered_to = email_service.send_prediction_email(
            to_email=to_email,
            plant_name=plant,
            disease_name=disease,
            confidence=confidence,
            ai_recommendation=rec,
            api_key=user_api_key
        )
        
        return jsonify({
            "success": success,
            "message": msg,
            "delivered_to": delivered_to,
            "prediction_id": pred_record_id,
            "plant": plant,
            "disease": disease
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/recommend", methods=["POST"])
@login_required
def get_recommend():
    data = request.get_json(silent=True) or {}
    plant = data.get("plant")
    disease = data.get("disease")
    if not plant or not disease:
        return jsonify({"success": False, "error": "Missing plant or disease"})
    
    recommendation = llm.get_recommendation(plant, disease)
    return jsonify({"success": True, "recommendation": recommendation})

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        
    if not message:
        return jsonify({"success": False, "error": "Missing message"})
        
    response = llm.chat_with_bot(session["session_id"], message, user_id=current_user.id)
    
    return jsonify({"success": True, "response": response})

@app.route("/history", methods=["GET"])
@login_required
def get_history():
    # Only get current user's history
    recent = db.get_recent_predictions(limit=20, user_id=current_user.id)
    
    chat_hist = []
    if "session_id" in session:
        chat_hist = db.get_chat_history(session["session_id"], limit=50, user_id=current_user.id)
        
    return jsonify({"success": True, "predictions": recent, "chat_history": chat_hist})

@app.route("/samples/<path:filename>")
def serve_sample(filename):
    samples_dir = os.path.join(BASE_DIR, "static", "samples")
    return send_from_directory(samples_dir, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
