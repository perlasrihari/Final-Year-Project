import os
import uuid
import functools
from datetime import datetime, timedelta
from collections import defaultdict

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, Response, send_from_directory, abort
)
from flask_login import (
    LoginManager, login_user, current_user,
    logout_user, login_required
)
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.utils import secure_filename
from sqlalchemy import or_

from config import Config
from database import db
from models import User, Property, Booking, Favorite, Review, PredictionResult
from forms import (
    LoginForm, RegistrationForm, PropertyForm, BookingForm, SearchForm,
    EditProfileForm, ChangePasswordForm, ReviewForm,
    RequestResetForm, ResetPasswordForm, PredictRentForm
)

import pandas as pd
import joblib
import folium

from flask_mail import Mail, Message

# Optional: Google OAuth (only if configured)
try:
    from flask_dance.contrib.google import make_google_blueprint, google
    GOOGLE_OAUTH_AVAILABLE = True
except Exception:
    GOOGLE_OAUTH_AVAILABLE = False


# ======================================================
# APP SETUP
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder="Frontend",
    static_folder="Frontend/assets",
    static_url_path="/assets",
)
app.config.from_object(Config)

# DB & Login
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# CSRF & Mail
csrf = CSRFProtect(app)
mail = Mail(app)

# Initialize database tables (including PredictionResult)
# If the configured DB is a local sqlite file, verify its header to avoid "file is not a database" errors.
with app.app_context():
    try:
        uri = app.config.get("SQLALCHEMY_DATABASE_URI", "") or ""
        if uri.startswith("sqlite:///"):
            # Resolve relative paths against multiple plausible locations (cwd, BASE_DIR, app.instance_path)
            db_rel = uri.replace("sqlite:///", "", 1)
            candidates = []
            # direct path
            candidates.append(db_rel)
            # relative to BASE_DIR (where app.py lives)
            candidates.append(os.path.join(BASE_DIR, db_rel))
            # relative to app.instance_path
            try:
                candidates.append(os.path.join(app.instance_path, os.path.basename(db_rel)))
            except Exception:
                pass
            # relative to current working directory
            candidates.append(os.path.join(os.getcwd(), db_rel))

            checked = False
            for db_path in candidates:
                try:
                    if os.path.exists(db_path):
                        with open(db_path, "rb") as f:
                            header = f.read(16)
                        checked = True
                        if header != b"SQLite format 3\x00":
                            backup_path = db_path + ".corrupt"
                            try:
                                os.replace(db_path, backup_path)
                                app.logger.warning(f"Detected corrupted sqlite DB. Renamed '{db_path}' -> '{backup_path}'")
                            except Exception as ex:
                                app.logger.error(f"Failed to back up corrupted DB '{db_path}': {ex}")
                            # only handle the first existing db file we find
                            break
                except Exception as ex:
                    app.logger.error(f"Error while checking sqlite DB file '{db_path}': {ex}")
            if not checked:
                app.logger.debug("No local sqlite DB file found among candidates; proceeding to create a new DB if necessary.")
        db.create_all()
    except Exception as e:
        app.logger.error(f"DB init error: {e}")

# File upload config
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ======================================================
# SECURITY HEADERS & ERROR HANDLERS
# ======================================================

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers[
        "Content-Security-Policy"
    ] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://code.jquery.com https://cdnjs.cloudflare.com https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com https://unpkg.com; "
        "img-src 'self' data:; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "connect-src 'self' localhost:* 127.0.0.1:*; "
        "form-action 'self'; "
        "frame-ancestors 'self'; "
        "base-uri 'self'"
    )
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    app.logger.error(f"CSRF Error: {e}")
    flash("Security token has expired or is invalid. Please try again.", "danger")
    return redirect(url_for("login"))


@app.errorhandler(413)
def request_entity_too_large(e):
    flash("The file you are trying to upload is too large. Maximum size is 10MB.", "danger")
    return redirect(request.url)


@app.errorhandler(404)
def page_not_found(e):
    return render_template(
        "error.html",
        error_code=404,
        error_title="Page Not Found",
        error_message="The page you are looking for does not exist.",
    ), 404


@app.errorhandler(403)
def forbidden(e):
    return render_template(
        "error.html",
        error_code=403,
        error_title="Forbidden",
        error_message="You do not have permission to access this resource.",
    ), 403


@app.errorhandler(500)
def internal_server_error(e):
    return render_template(
        "error.html",
        error_code=500,
        error_title="Internal Server Error",
        error_message="Something went wrong on our end. Please try again later.",
    ), 500


# ======================================================
# RATE LIMITING (LOGIN)
# ======================================================

login_attempts = defaultdict(list)

def rate_limit(max_attempts=10, window_seconds=600):
    def decorator(f):
        @functools.wraps(f)
        def wrapped_view(*args, **kwargs):
            ip = request.remote_addr or "unknown"
            now = datetime.now()
            login_attempts[ip] = [
                t for t in login_attempts[ip]
                if (now - t) < timedelta(seconds=window_seconds)
            ]
            if len(login_attempts[ip]) >= max_attempts:
                return render_template(
                    "error.html",
                    error_code=429,
                    error_title="Too Many Attempts",
                    error_message=f"Too many login attempts. Please try again after {window_seconds // 60} minutes.",
                ), 429
            login_attempts[ip].append(now)
            return f(*args, **kwargs)
        return wrapped_view
    return decorator


# ======================================================
# MODEL & DATASET LOADING
# ======================================================

# These files live in the same folder as app.py
MODEL_PATH = os.path.join(BASE_DIR, "house_rent_model.pkl")
UI_DATASET_PATH = os.path.join(BASE_DIR, "House_Rent_10k_major_cities.csv")
DATASET_PATH = os.getenv("DATASET_PATH", UI_DATASET_PATH)

model = None
ui_df = None

def load_ui_dataset(path: str) -> None:
    global ui_df
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            ui_df = df
            app.logger.info(f"UI dataset loaded from {path}")
        else:
            ui_df = None
            app.logger.warning(f"Dataset not found at {path}")
    except Exception as e:
        ui_df = None
        app.logger.error(f"Dataset load error: {e}")

def get_model():
    """Lazy-load and cache the ML model. Returns the model or None."""
    global model
    if model is not None:
        return model
    try:
        if os.path.exists(MODEL_PATH):
            import time
            t0 = time.time()
            model = joblib.load(MODEL_PATH)
            app.logger.info(f"ML model loaded successfully in {time.time() - t0:.2f}s.")
        else:
            app.logger.warning(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        app.logger.error(f"Model load error: {e}")
        model = None
    return model

# Load dataset at startup; model will be loaded lazily when needed
load_ui_dataset(DATASET_PATH)


# ======================================================
# LOGIN MANAGER
# ======================================================

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None


# ======================================================
# GOOGLE OAUTH (OPTIONAL)
# ======================================================

if GOOGLE_OAUTH_AVAILABLE:
    google_bp = make_google_blueprint(
        client_id=os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET"),
        redirect_to="google_login",
    )
    app.register_blueprint(google_bp, url_prefix="/login")

    @app.route("/google_login")
    def google_login():
        if not google.authorized:
            return redirect(url_for("google.login"))
        resp = google.get("/oauth2/v2/userinfo")
        user_info = resp.json()
        email = user_info["email"]
        name = user_info.get("name", email.split("@")[0])
        user = User.query.filter_by(email=email).first()
        if not user:
            from secrets import token_urlsafe
            user = User(name=name, email=email, role="customer", verified=True)
            user.set_password(token_urlsafe(16))
            db.session.add(user)
            db.session.commit()
        login_user(user)
        flash("Logged in with Google!", "success")
        return redirect(url_for("dashboard"))


# ======================================================
# HELPERS
# ======================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message("Password Reset Request", recipients=[user.email])
    msg.body = f"""
To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
"""
    mail.send(msg)


def format_inr(value):
    """Formats a number in Indian currency format (e.g., 1,00,000)."""
    if not isinstance(value, (int, float)):
        return value
    s = str(int(value))[::-1]
    groups = []
    i = 0
    while i < len(s):
        if i == 0:
            groups.append(s[i:i+3])
            i += 3
        else:
            groups.append(s[i:i+2])
            i += 2
    return "₹" + ",".join(groups)[::-1].strip(",")

app.jinja_env.filters["inr"] = format_inr


def infer_built_type(bedrooms, bathrooms, area_type):
    try:
        b = int(bedrooms or 0)
    except Exception:
        b = 0
    at = (area_type or "").strip()
    if at.lower() == "plot area":
        return "Plot"
    if b <= 1:
        return "Apartment"
    if b == 2:
        return "Apartment"
    if b == 3:
        return "Independent House"
    if b >= 4:
        return "Villa"
    return "Apartment"

def derive_pid_from_row(row: dict) -> str:
    try:
        import hashlib
        base = f"{row.get('City','')}-{row.get('BHK','')}-{row.get('Bathroom','')}-{row.get('Size','')}-{row.get('Rent','')}"
        digest = hashlib.md5(base.encode('utf-8')).hexdigest()
        num = int(digest[:8], 16) % 100000000
        return f"TH{str(num).zfill(8)}"
    except Exception:
        return "TH00000000"


def enrich_from_dataset(payload: dict) -> dict:
    try:
        if ui_df is None or ui_df.empty:
            return payload
        city = str(payload.get("city") or payload.get("location") or "")
        bhk = int(payload.get("bedrooms") or 0)
        bath = int(payload.get("bathrooms") or 0)
        size = float(payload.get("area") or payload.get("size") or 0)
        rent = float(payload.get("price") or 0)

        df = ui_df.copy()
        if city:
            try:
                df = df[df["City"].astype(str) == city]
            except Exception:
                pass
        if df.empty:
            df = ui_df.copy()

        def score(row):
            s = 0.0
            try:
                s += abs(float(row.get("Size", 0) or 0) - size) / 1000.0
            except Exception:
                pass
            try:
                s += 0.5 * abs(float(row.get("Rent", 0) or 0) - rent) / max(rent or 1.0, 1.0)
            except Exception:
                pass
            try:
                s += 0.2 * abs(int(row.get("BHK", 0) or 0) - bhk)
            except Exception:
                pass
            try:
                s += 0.2 * abs(int(row.get("Bathroom", 0) or 0) - bath)
            except Exception:
                pass
            return s

        best_row = None
        best_score = None
        for row in df.to_dict("records"):
            sc = score(row)
            if best_score is None or sc < best_score:
                best_score = sc
                best_row = row

        if best_row:
            payload.setdefault("area_type", str(best_row.get("Area Type", "")))
            if not payload.get("area_type"):
                payload["area_type"] = str(best_row.get("Area Type", ""))
            if not payload.get("locality"):
                payload["locality"] = str(best_row.get("Area Locality", ""))
            if not payload.get("contact"):
                payload["contact"] = str(best_row.get("Point of Contact", ""))
            if not payload.get("pid"):
                payload["pid"] = str(best_row.get("Property ID") or best_row.get("Property_ID") or best_row.get("Post ID") or derive_pid_from_row(best_row))
        return payload
    except Exception:
        return payload


# ======================================================
# STATIC & VITE
# ======================================================

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory("Frontend/assets", filename)


@app.route("/property_pics/<path:filename>")
def serve_property_pic(filename):
    return send_from_directory(os.path.join(app.root_path, "static", "property_pics"), filename)


@app.route("/@vite/client", methods=["GET"])
def handle_vite_client():
    return Response("", mimetype="application/javascript")


# ======================================================
# PUBLIC PAGES (HOME / ABOUT / CONTACT / LISTING)
# ======================================================

@app.route("/")
def index():
    featured = []
    home_cities = []
    popular_cities = []
    property_types = []

    if ui_df is not None and not ui_df.empty:
        df = ui_df.copy()

        # Featured properties
        featured = df.sample(min(6, len(df))).to_dict("records")

        # Cities and counts
        if "City" in df.columns:
            city_counts = df["City"].dropna().astype(str).value_counts()
            home_cities = [{"name": c, "count": int(n)} for c, n in city_counts.items()]
            popular_cities = [{"name": c, "count": int(n)} for c, n in city_counts.head(40).items()]

        # Property types from built type (derived), not area type
        try:
            bt_series = df.apply(lambda r: infer_built_type(r.get("BHK"), r.get("Bathroom"), r.get("Area Type")), axis=1)
            bt_counts = bt_series.dropna().astype(str).value_counts()
            property_types = [{"name": name, "count": int(cnt)} for name, cnt in bt_counts.items()]
        except Exception:
            property_types = []

    return render_template(
        "index.html",
        featured_properties=featured,
        home_cities=home_cities,
        popular_cities=popular_cities,
        property_types=property_types,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # In a real application, you would process the form data here
        # For example, send an email or save to a database
        flash("Your message has been sent successfully!", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/listing")
def listing():
    import random

    db_props = Property.query.order_by(Property.created_at.desc()).all()

    properties = []
    if db_props:
        for p in db_props:
            properties.append({
                "id": p.id,
                "title": p.title or f"Property in {p.city}",
                "price": float(p.price) if p.price is not None else 0.0,
                "location": p.city or "",
                "description": p.description or "",
                "address": getattr(p, "address", ""),
                "category": infer_built_type(p.bedrooms, p.bathrooms, p.area_type),
                "bedrooms": p.bedrooms or 0,
                "bathrooms": p.bathrooms or 0,
                "area": p.size or 0,
                "furnishing_status": p.furnishing_status or "",
                "tenant_preferred": p.tenant_preferred or "",
                "area_type": p.area_type or "",
                "locality": getattr(p, "locality", ""),
                "contact": getattr(p, "point_of_contact", ""),
                "pid": getattr(p, "property_id", None),
                "created_at": (p.created_at.isoformat() if p.created_at else ""),
                "image_url": (url_for("serve_property_pic", filename=p.image_file) if p.image_file else None),
            })
    else:
        # Fallback to dataset if DB has no properties
        ds = ui_df.to_dict("records") if (ui_df is not None and not ui_df.empty) else []

        image_dir = os.path.join(BASE_DIR, "Frontend", "assets", "img", "property")
        image_files = []
        if os.path.isdir(image_dir):
            image_files = [
                f for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))
            ]

        for row in ds:
            properties.append({
                "id": None,
                "title": f"{int(row.get('BHK', 0))}-BHK in {row.get('City', '')}",
                "price": float(row.get("Rent", 0) or 0),
                "location": str(row.get("City", "")),
                "description": str(row.get("Description", "")),
                "address": str(row.get("Address", "")),
                "category": infer_built_type(row.get("BHK", 0), row.get("Bathroom", 0), row.get("Area Type", "")),
                "bedrooms": int(row.get("BHK", 0) or 0),
                "bathrooms": int(row.get("Bathroom", 0) or 0),
                "area": float(row.get("Size", 0) or 0),
                "furnishing_status": str(row.get("Furnishing Status", "")),
                "tenant_preferred": str(row.get("Tenant Preferred", "")),
                "area_type": str(row.get("Area Type", "")),
                "locality": str(row.get("Area Locality", "")),
                "contact": str(row.get("Point of Contact", "")),
                "pid": str(row.get("Property ID") or row.get("Property_ID") or row.get("Post ID") or derive_pid_from_row(row)),
                "created_at": "",
                "image_url": (
                    url_for("static", filename=f"img/property/{random.choice(image_files)}")
                    if image_files else None
                ),
            })

    return render_template("listing.html", properties=properties)


 

@app.route("/property_preview")
def property_preview():
    args = request.args
    property_payload = {
        "id": None,
        "title": args.get("title") or "Property",
        "price": float(args.get("price") or 0),
        "bedrooms": int(args.get("bedrooms") or 0),
        "bathrooms": int(args.get("bathrooms") or 0),
        "area": float(args.get("area") or 0),
        "category": args.get("category") or "Property",
        "address": args.get("address") or "",
        "city": args.get("city") or "",
        "description": args.get("description") or "",
        "furnishing_status": args.get("furnishing_status") or "",
        "tenant_preferred": args.get("tenant_preferred") or "",
        "pid": args.get("pid") or "",
        "area_type": args.get("area_type") or "",
        "locality": args.get("locality") or "",
        "contact": args.get("contact") or "",
        "image_url": None,
        "owner": {
            "name": "Listing Agent",
            "role": "Agent",
            "phone": "",
        },
    }
    property_payload = enrich_from_dataset(property_payload)
    featured = []
    try:
        import random
        ds = ui_df.to_dict("records") if (ui_df is not None and not ui_df.empty) else []
        city = property_payload["city"]
        pool = [r for r in ds if (city and str(r.get("City",""))==city)] or ds
        for row in pool[:3]:
            title = f"{int(row.get('BHK',0))}-BHK in {row.get('City','')}"
            url = url_for("property_preview", title=title, price=float(row.get("Rent",0) or 0), bedrooms=int(row.get("BHK",0) or 0), bathrooms=int(row.get("Bathroom",0) or 0), area=float(row.get("Size",0) or 0), category=str(row.get("Area Type","")), address=str(row.get("Address","")), city=str(row.get("City","")), furnishing_status=str(row.get("Furnishing Status","")), tenant_preferred=str(row.get("Tenant Preferred","")), pid=str(row.get("Property ID") or row.get("Property_ID") or derive_pid_from_row(row)), area_type=str(row.get("Area Type","")), locality=str(row.get("Area Locality","")), contact=str(row.get("Point of Contact","")))
            featured.append({
                "title": title,
                "city": str(row.get("City","")),
                "price": float(row.get("Rent",0) or 0),
                "image_url": url_for("serve_assets", filename=f"img/property/featured-grid{random.randint(1,6)}.jpg"),
                "url": url,
            })
    except Exception:
        pass
    return render_template("listing-details.html", property=property_payload, featured_properties=featured)

# ======================================================
# RENT PREDICTION (PAGE + API)
# ======================================================

@app.route("/rent_prediction", methods=["GET", "POST"])
def rent_prediction():
    form = PredictRentForm()
    prediction_result = None

    # Dynamic options from dataset for dropdowns
    city_options = []
    furnishing_options = []
    tenant_options = []
    area_type_options = []
    contact_options = []
    localities_by_city = {}

    if ui_df is not None and not ui_df.empty:
        try:
            city_options = sorted(set(ui_df["City"].dropna().astype(str).tolist()))
        except Exception:
            city_options = []
        try:
            furnishing_options = sorted(set(ui_df["Furnishing Status"].dropna().astype(str).tolist()))
        except Exception:
            furnishing_options = ["Unfurnished", "Semi-Furnished", "Furnished"]
        try:
            tenant_options = sorted(set(ui_df["Tenant Preferred"].dropna().astype(str).tolist()))
        except Exception:
            tenant_options = ["Bachelors", "Family", "Bachelors/Family"]
        try:
            area_type_options = sorted(set(ui_df["Area Type"].dropna().astype(str).tolist()))
        except Exception:
            area_type_options = ["Super Area", "Carpet Area", "Built Area", "Plot Area"]
        try:
            contact_options = sorted(set(ui_df["Point of Contact"].dropna().astype(str).tolist()))
        except Exception:
            contact_options = ["Contact Owner", "Contact Agent", "Contact Builder"]
        try:
            if "Area Locality" in ui_df.columns:
                for city, sub in ui_df.groupby("City"):
                    localities_by_city[str(city)] = sorted(set(sub["Area Locality"].dropna().astype(str).tolist()))
        except Exception:
            localities_by_city = {}

    # Set choices for validation
    form.city.choices = [(c, c) for c in city_options]
    form.furnishing_status.choices = [(f, f) for f in furnishing_options]
    form.tenant_preferred.choices = [(t, t) for t in tenant_options]
    form.area_type.choices = [(a, a) for a in area_type_options]

    if form.validate_on_submit():
        mdl = get_model()
        if mdl is None:
            flash("Prediction model is not available.", "danger")
        else:
            try:
                # Use field names consistent with your original PredictRentForm
                data = {
                    "Size": [form.size.data],
                    "BHK": [form.bhk.data],
                    "Bathroom": [form.bathroom.data],
                    "City": [form.city.data],
                    "Furnishing Status": [form.furnishing_status.data],
                    "Tenant Preferred": [form.tenant_preferred.data],
                    "Area Type": [form.area_type.data],
                }
                input_df = pd.DataFrame(data)
                predicted = mdl.predict(input_df)[0]
                prediction_result = f"₹{predicted:,.0f}"
            except Exception as e:
                app.logger.error(f"Prediction error: {e}")
                flash("An error occurred during prediction. Please try again.", "danger")

    return render_template(
        "rent-prediction.html",
        form=form,
        prediction_result=prediction_result,
        city_options=city_options,
        furnishing_options=furnishing_options,
        tenant_options=tenant_options,
        area_type_options=area_type_options,
        contact_options=contact_options,
        localities_by_city=localities_by_city,
    )

# ======================================================
# Legacy .html URL mappings (for static template links)
# ======================================================

@app.route("/<page>.html")
def legacy_html(page):
    """Support existing hardcoded links to .html by redirecting to routes
    or rendering the template directly when appropriate.
    """
    route_map = {
        "index": "index",
        "listing": "listing",
        "rent-prediction": "rent_prediction",
        "login": "login",
        "sign-up": "signup",
        "about": "about",
        "contact": "contact",
        "bookings": "bookings",
        "favorites": "favorites",
        "dashboard": "dashboard",
        "owner-dashboard": "owner_dashboard",
        "my-properties": "owner_dashboard",
        "profile": "profile",
        "settings": "settings",
    }

    if page in route_map:
        return redirect(url_for(route_map[page]))

    # Fallback: try to render the template if it exists
    template_name = f"{page}.html"
    try:
        return render_template(template_name)
    except Exception:
        abort(404)


@app.route("/predict_rent", methods=["POST"])
@csrf.exempt  # if called from JS without CSRF token
def predict_rent_api():
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    data = request.get_json() or {}
    form = PredictRentForm(meta={'csrf': False}, data=data)

    # Prepare choices for SelectFields
    if ui_df is not None and not ui_df.empty:
        city_options = sorted(set(ui_df["City"].dropna().astype(str).tolist()))
        furnishing_options = sorted(set(ui_df["Furnishing Status"].dropna().astype(str).tolist()))
        tenant_options = sorted(set(ui_df["Tenant Preferred"].dropna().astype(str).tolist()))
        area_type_options = sorted(set(ui_df["Area Type"].dropna().astype(str).tolist()))
    else:
        city_options = []
        furnishing_options = ["Unfurnished", "Semi-Furnished", "Furnished"]
        tenant_options = ["Bachelors", "Family", "Bachelors/Family"]
        area_type_options = ["Super Area", "Carpet Area", "Built Area", "Plot Area"]

    form.city.choices = [(c, c) for c in city_options]
    form.furnishing_status.choices = [(f, f) for f in furnishing_options]
    form.tenant_preferred.choices = [(t, t) for t in tenant_options]
    form.area_type.choices = [(a, a) for a in area_type_options]

    if ui_df is not None and not ui_df.empty:
        city_options = sorted(set(ui_df["City"].dropna().astype(str).tolist()))
        furnishing_options = sorted(set(ui_df["Furnishing Status"].dropna().astype(str).tolist()))
        tenant_options = sorted(set(ui_df["Tenant Preferred"].dropna().astype(str).tolist()))
        area_type_options = sorted(set(ui_df["Area Type"].dropna().astype(str).tolist()))
        # Area locality choices union for validation (optional)
        try:
            all_localities = sorted(set(ui_df["Area Locality"].dropna().astype(str).tolist()))
        except Exception:
            all_localities = []
    else:
        city_options = []
        furnishing_options = ["Unfurnished", "Semi-Furnished", "Furnished"]
        tenant_options = ["Bachelors", "Family", "Bachelors/Family"]
        area_type_options = ["Super Area", "Carpet Area", "Built Area", "Plot Area"]
        all_localities = []

    form.city.choices = [(c, c) for c in city_options]
    form.furnishing_status.choices = [(f, f) for f in furnishing_options]
    form.tenant_preferred.choices = [(t, t) for t in tenant_options]
    form.area_type.choices = [(a, a) for a in area_type_options]
    if hasattr(form, 'area_locality'):
        form.area_locality.choices = [(l, l) for l in all_localities]

    if not form.validate():
        return jsonify({"error": "Invalid form submission", "errors": form.errors}), 400

    try:
        input_df = pd.DataFrame({
            "Size": [form.size.data],
            "BHK": [form.bhk.data],
            "Bathroom": [form.bathroom.data],
            "City": [form.city.data],
            "Furnishing Status": [form.furnishing_status.data],
            "Tenant Preferred": [form.tenant_preferred.data],
            "Area Type": [form.area_type.data],
        })
        predicted = model.predict(input_df)[0]

        matching_properties = []
        if ui_df is not None and not ui_df.empty and "Rent" in ui_df.columns:
            min_price = predicted * 0.9
            max_price = predicted * 1.1
            df_match = ui_df[
                (ui_df["Rent"] >= min_price) & (ui_df["Rent"] <= max_price)
            ]
            if form.city.data and form.city.data != "Any":
                df_match = df_match[df_match["City"] == form.city.data]
            matching_properties = df_match.head(50).to_dict("records")

        return jsonify({
            "predicted_rent": f"{predicted:.2f}",
            "matching_properties": matching_properties,
        })
    except Exception as e:
        app.logger.error(f"Prediction API error: {e}")
        return jsonify({"error": "Internal error"}), 500


# ======================================================
# DATASET IMPORT
# ======================================================

def import_properties_from_df(df, limit: int = 500):
    admin_email = app.config.get("ADMIN_EMAIL", "admin@example.com")
    admin_user = User.query.filter_by(email=admin_email).first()
    if not admin_user:
        return 0
    required = {"Rent", "Size", "BHK", "Bathroom", "City"}
    if not required.issubset(set(df.columns)):
        return 0
    count = 0
    for _, row in df.head(limit).iterrows():
        try:
            prop = Property(
                title=f"{int(row['BHK']) if pd.notnull(row['BHK']) else 0}-BHK in {row['City']}",
                description=str(row.get("Description", "")) or "",
                address=str(row.get("Address", row["City"])) or str(row["City"]),
                city=str(row["City"]),
                price=float(row["Rent"]) if pd.notnull(row["Rent"]) else 0.0,
                bedrooms=int(row["BHK"]) if pd.notnull(row["BHK"]) else None,
                bathrooms=int(row["Bathroom"]) if pd.notnull(row["Bathroom"]) else None,
                size=float(row["Size"]) if pd.notnull(row["Size"]) else None,
                furnishing_status=str(row.get("Furnishing Status", "")) or None,
                tenant_preferred=str(row.get("Tenant Preferred", "")) or None,
                area_type=str(row.get("Area Type", "")) or None,
                owner_id=admin_user.id,
            )
            db.session.add(prop)
            count += 1
        except Exception:
            db.session.rollback()
    db.session.commit()
    return count


@app.route("/admin/import_dataset", methods=["GET", "POST"])
@login_required
def admin_import_dataset():
    if current_user.role != "admin":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))
    filename = request.values.get("filename") or DATASET_PATH
    to_db = str(request.values.get("to_db", "0")).lower() in {"1", "true", "yes"}
    if not filename or not os.path.exists(filename):
        flash("Dataset file not found", "danger")
        return redirect(url_for("admin_dashboard"))
    load_ui_dataset(filename)
    flash("UI dataset loaded", "success")
    if ui_df is not None and to_db:
        imported = import_properties_from_df(ui_df)
        flash(f"Imported {imported} properties into the database", "success")
    return redirect(url_for("admin_dashboard"))


# ======================================================
# AUTH: LOGIN / REGISTER / SIGNUP / LOGOUT
# ======================================================

class SimpleRegistrationForm(RegistrationForm):
    """Use your existing RegistrationForm but without strict email validator (if needed)."""
    pass

class SimpleLoginForm(LoginForm):
    """Use your existing LoginForm but without strict email validator (if needed)."""
    pass


@app.route("/login", methods=["GET", "POST"])
@rate_limit(max_attempts=5, window_seconds=300)
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    form = SimpleLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_attempts[request.remote_addr] = []
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Login Unsuccessful. Please check email and password", "danger")
    return render_template("login.html", title="Login", form=form)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    form = SimpleRegistrationForm()
    if form.validate_on_submit():
        user = User(
            name=form.name.data,
            email=form.email.data,
            role=form.role.data,
            phone=form.phone.data,
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Your account has been created! You are now able to log in", "success")
        return redirect(url_for("login"))
    return render_template("sign-up.html", title="Register", form=form)


# 🔥 This fixes the current error in your logs
@app.route("/sign-up")
def signup():
    # Navbar uses url_for('signup'), so we redirect to the register page
    return redirect(url_for("register"))


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))


# ======================================================
# DASHBOARDS
# ======================================================

@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    elif current_user.role == "owner":
        return redirect(url_for("owner_dashboard"))
    else:
        return redirect(url_for("customer_dashboard"))


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))
    properties = Property.query.all()
    bookings = Booking.query.all()
    users = User.query.all()
    return render_template(
        "owner/owner-dashboard.html",
        properties=properties,
        bookings=bookings,
        users=users,
        reviews=Review.query.all(),
    )


@app.route("/owner/dashboard")
@login_required
def owner_dashboard():
    if current_user.role != "owner":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))
    properties = Property.query.filter_by(owner_id=current_user.id).all()
    # Get all property IDs owned by the current user
    property_ids = [prop.id for prop in properties]
    # Fetch all bookings related to those properties
    bookings = Booking.query.filter(Booking.property_id.in_(property_ids)).order_by(Booking.created_at.desc()).all()
    # Reviews for owner's properties
    reviews = Review.query.filter(Review.property_id.in_(property_ids)).all()
    # Earnings from confirmed bookings
    earnings_total = sum((b.property.price or 0) for b in bookings if b.status == "confirmed")
    # Average rating across all owner's properties
    avg_rating = round(sum(r.rating for r in reviews) / len(reviews), 1) if reviews else 0
    # Recent properties
    recent_properties = Property.query.filter_by(owner_id=current_user.id).order_by(Property.created_at.desc()).limit(6).all()
    # Additional metrics
    active_listings = sum(1 for p in properties if p.available)
    pending_bookings = sum(1 for b in bookings if b.status == "pending")
    confirmed_bookings = sum(1 for b in bookings if b.status == "confirmed")
    # Monthly earnings (confirmed bookings in current month)
    from datetime import datetime
    now = datetime.utcnow()
    monthly_earnings = sum((b.property.price or 0) for b in bookings if b.status == "confirmed" and b.created_at.year == now.year and b.created_at.month == now.month)
    # Favorites on owner's properties
    favorites_count = Favorite.query.join(Property, Favorite.property_id == Property.id).filter(Property.owner_id == current_user.id).count()

    return render_template(
        "owner/owner-dashboard.html",
        properties=properties,
        bookings=bookings,
        reviews=reviews,
        earnings_total=earnings_total,
        avg_rating=avg_rating,
        recent_properties=recent_properties,
        active_listings=active_listings,
        pending_bookings=pending_bookings,
        confirmed_bookings=confirmed_bookings,
        monthly_earnings=monthly_earnings,
        favorites_count=favorites_count,
    )


@app.route("/customer/dashboard", methods=["GET", "POST"])
@login_required
def customer_dashboard():
    if current_user.role != "customer":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))

    form = SearchForm()
    properties_query = Property.query

    if form.validate_on_submit():
        try:
            if form.location.data:
                properties_query = properties_query.filter(
                    Property.city.ilike(f"%{form.location.data}%")
                )
            if form.min_price.data is not None:
                properties_query = properties_query.filter(
                    Property.price >= form.min_price.data
                )
            if form.max_price.data is not None:
                properties_query = properties_query.filter(
                    Property.price <= form.max_price.data
                )
            if form.bedrooms.data is not None:
                properties_query = properties_query.filter(
                    Property.bedrooms >= form.bedrooms.data
                )
            if form.bathrooms.data is not None:
                properties_query = properties_query.filter(
                    Property.bathrooms >= form.bathrooms.data
                )
            if form.min_size.data is not None:
                properties_query = properties_query.filter(
                    Property.size >= form.min_size.data
                )
            if form.max_size.data is not None:
                properties_query = properties_query.filter(
                    Property.size <= form.max_size.data
                )
            if form.furnishing_status.data:
                properties_query = properties_query.filter(
                    Property.furnishing_status == form.furnishing_status.data
                )
            if form.tenant_preferred.data:
                properties_query = properties_query.filter(
                    Property.tenant_preferred == form.tenant_preferred.data
                )
        except Exception as e:
            app.logger.error(f"Search error: {e}")
            flash("An error occurred while processing your search.", "danger")

    properties = properties_query.all()

    bookings_count = Booking.query.filter_by(customer_id=current_user.id).count()
    favorites_count = Favorite.query.filter_by(user_id=current_user.id).count()
    predictions_count = PredictionResult.query.filter_by(user_id=current_user.id).count()

    # Default map on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for prop in properties:
        if prop.latitude and prop.longitude:
            folium.Marker(
                [prop.latitude, prop.longitude],
                popup=f"<b>{prop.title}</b><br>₹{prop.price}/month<br>"
                      f"<a href='{url_for('customer_property_detail', property_id=prop.id)}'>View</a>",
                tooltip=prop.title,
            ).add_to(m)
    map_html = m._repr_html_()

    return render_template(
        "customer/dashboard.html",
        form=form,
        properties=properties,
        map_html=map_html,
        bookings_count=bookings_count,
        favorites_count=favorites_count,
        predictions_count=predictions_count,
    )


# ======================================================
# OWNER: PROPERTY CRUD
# ======================================================

@app.route("/owner/add_property", methods=["GET", "POST"])
@login_required
def add_property():
    if current_user.role != "owner":
        flash("You are not authorized to access this page.", "danger")
        return redirect(url_for("dashboard"))
    form = PropertyForm()
    if form.validate_on_submit():
        prop = Property(
            title=form.title.data,
            description=form.description.data,
            address=form.address.data,
            city=form.city.data,
            price=form.price.data,
            bedrooms=form.bedrooms.data,
            bathrooms=form.bathrooms.data,
            latitude=form.latitude.data,
            longitude=form.longitude.data,
            owner_id=current_user.id,
        )
        db.session.add(prop)
        db.session.commit()
        flash("Property added successfully!", "success")
        return redirect(url_for("owner_dashboard"))
    return render_template("owner/create-listing.html", form=form)


@app.route("/owner/edit_property/<int:property_id>", methods=["GET", "POST"])
@login_required
def edit_property(property_id):
    prop = Property.query.get_or_404(property_id)
    if prop.owner_id != current_user.id:
        flash("You are not authorized to edit this property", "danger")
        return redirect(url_for("owner_dashboard"))

    form = PropertyForm(obj=prop)
    if form.validate_on_submit():
        prop.title = form.title.data
        prop.description = form.description.data
        prop.address = form.address.data
        prop.city = form.city.data
        prop.price = form.price.data
        prop.bedrooms = form.bedrooms.data
        prop.bathrooms = form.bathrooms.data
        prop.latitude = form.latitude.data
        prop.longitude = form.longitude.data

        # Image upload
        image_file = form.image_file.data
        if image_file and image_file.filename:
            if not allowed_file(image_file.filename):
                flash("Invalid file type. Only jpg, jpeg, and png are allowed.", "danger")
                return render_template("owner/create-listing.html", form=form, property=prop)

            if prop.image_file and prop.image_file != "default.jpg":
                old_path = os.path.join(app.root_path, "static", "property_pics", prop.image_file)
                if os.path.exists(old_path):
                    os.remove(old_path)

            filename = secure_filename(image_file.filename)
            filename = f"{uuid.uuid4().hex}_{filename}"
            save_dir = os.path.join(app.root_path, "static", "property_pics")
            os.makedirs(save_dir, exist_ok=True)
            image_path = os.path.join(save_dir, filename)
            image_file.save(image_path)
            prop.image_file = filename

        db.session.commit()
        flash("Property updated successfully!", "success")
        return redirect(url_for("owner_dashboard"))

    return render_template("owner/create-listing.html", form=form, property=prop)


@app.route("/owner/delete_property/<int:property_id>", methods=["POST"])
@login_required
def delete_property(property_id):
    prop = Property.query.get_or_404(property_id)
    if prop.owner_id != current_user.id:
        flash("You are not authorized to delete this property", "danger")
        return redirect(url_for("owner_dashboard"))

    Booking.query.filter_by(property_id=prop.id).delete()
    db.session.delete(prop)
    db.session.commit()
    flash("Property deleted successfully", "success")
    return redirect(url_for("owner_dashboard"))


@app.route("/owner/property/<int:property_id>")
@login_required
def owner_property_detail(property_id):
    prop = Property.query.get_or_404(property_id)
    if prop.owner_id != current_user.id:
        flash("You are not authorized to view this property", "danger")
        return redirect(url_for("owner_dashboard"))
    bookings = Booking.query.filter_by(property_id=prop.id).all()
    return render_template("listing-details.html", property=prop, bookings=bookings)


@app.route("/owner/booking/<int:booking_id>/confirm", methods=["POST"])
@login_required
def confirm_booking(booking_id):
    booking = Booking.query.get_or_404(booking_id)
    prop = Property.query.get_or_404(booking.property_id)
    if prop.owner_id != current_user.id:
        flash("You are not authorized to confirm this booking", "danger")
        return redirect(url_for("owner_dashboard"))
    booking.status = "confirmed"
    db.session.commit()
    flash("Booking confirmed!", "success")
    return redirect(url_for("owner_property_detail", property_id=prop.id))


@app.route("/owner/booking/<int:booking_id>/reject", methods=["POST"])
@login_required
def reject_booking(booking_id):
    booking = Booking.query.get_or_404(booking_id)
    prop = Property.query.get_or_404(booking.property_id)
    if prop.owner_id != current_user.id:
        flash("You are not authorized to reject this booking", "danger")
        return redirect(url_for("owner_dashboard"))
    booking.status = "rejected"
    db.session.commit()
    flash("Booking rejected", "info")
    return redirect(url_for("owner_property_detail", property_id=prop.id))


# ======================================================
# CUSTOMER: PROPERTY VIEW / BOOKINGS / FAVORITES / REVIEWS
# ======================================================

@app.route("/customer/property/<int:property_id>", methods=["GET", "POST"])
@login_required
def customer_property_detail(property_id):
    prop = Property.query.get_or_404(property_id)
    form = BookingForm()
    if form.validate_on_submit():
        overlapping = Booking.query.filter(
            or_(
                (Booking.start_date <= form.start_date.data) & (Booking.end_date >= form.start_date.data),
                (Booking.start_date <= form.end_date.data) & (Booking.end_date >= form.end_date.data),
                (Booking.start_date >= form.start_date.data) & (Booking.end_date <= form.end_date.data),
            )
        ).first()
        if overlapping:
            flash("This property is already booked for the selected dates", "danger")
        else:
            booking = Booking(
                start_date=form.start_date.data,
                end_date=form.end_date.data,
                property_id=prop.id,
                customer_id=current_user.id,
                status="pending",
            )
            db.session.add(booking)
            db.session.commit()
            flash("Booking request sent! The owner will confirm soon.", "success")
            return redirect(url_for("customer_bookings"))

    return render_template("listing-details.html", property=prop, form=form)


@app.route("/customer/bookings")
@login_required
def customer_bookings():
    if current_user.role != "customer":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))
    bookings = Booking.query.filter_by(customer_id=current_user.id).all()
    return render_template("customer/bookings.html", bookings=bookings)


@app.route("/customer/booking/<int:booking_id>/cancel", methods=["POST"])
@login_required
def cancel_booking(booking_id):
    booking = Booking.query.get_or_404(booking_id)
    if booking.customer_id != current_user.id:
        flash("You are not authorized to cancel this booking", "danger")
        return redirect(url_for("customer_bookings"))
    if booking.status == "confirmed":
        flash("This booking is already confirmed and cannot be canceled", "warning")
    else:
        db.session.delete(booking)
        db.session.commit()
        flash("Booking canceled successfully", "success")
    return redirect(url_for("customer_bookings"))


@app.route("/property/<int:property_id>")
def property_detail(property_id):
    prop = Property.query.get_or_404(property_id)
    review_form = ReviewForm()
    reviews = Review.query.filter_by(property_id=property_id).order_by(Review.created_at.desc()).all()
    is_favorited = False
    if current_user.is_authenticated:
        fav = Favorite.query.filter_by(user_id=current_user.id, property_id=property_id).first()
        is_favorited = fav is not None
    featured = []
    try:
        q = Property.query
        if prop.city:
            q = q.filter(Property.city == prop.city)
        featured_q = q.filter(Property.id != property_id).order_by(Property.created_at.desc()).limit(3).all()
        for fp in featured_q:
            featured.append({
                "title": fp.title or f"{fp.bedrooms or ''}-BHK in {fp.city or ''}",
                "city": fp.city or "",
                "price": float(fp.price or 0),
                "image_url": (url_for("serve_property_pic", filename=fp.image_file) if fp.image_file else url_for("serve_assets", filename="img/property/featured-grid1.jpg")),
                "url": url_for("property_detail", property_id=fp.id),
            })
    except Exception:
        pass
    # Enrich missing fields from dataset for display
    try:
        payload = {
            "id": prop.id,
            "title": prop.title,
            "price": float(prop.price or 0),
            "bedrooms": int(prop.bedrooms or 0),
            "bathrooms": int(prop.bathrooms or 0),
            "area": float(prop.size or 0),
            "category": infer_built_type(prop.bedrooms, prop.bathrooms, prop.area_type),
            "address": prop.address or "",
            "city": prop.city or "",
            "description": prop.description or "",
            "furnishing_status": prop.furnishing_status or "",
            "tenant_preferred": prop.tenant_preferred or "",
            "pid": getattr(prop, "property_id", None) or "",
            "area_type": prop.area_type or "",
            "locality": getattr(prop, "locality", ""),
            "contact": getattr(prop, "point_of_contact", ""),
        }
        payload = enrich_from_dataset(payload)
        # Assign transient attributes so template access via prop works
        try:
            prop.pid = payload.get("pid")
            prop.locality = payload.get("locality")
            prop.point_of_contact = payload.get("contact")
            if not prop.area_type:
                prop.area_type = payload.get("area_type")
        except Exception:
            pass
    except Exception:
        pass

    return render_template(
        "listing-details.html",
        property=prop,
        review_form=review_form,
        reviews=reviews,
        is_favorited=is_favorited,
        featured_properties=featured,
    )


@app.route("/property/<int:property_id>/review", methods=["POST"])
@login_required
def add_review(property_id):
    prop = Property.query.get_or_404(property_id)
    form = ReviewForm()
    if form.validate_on_submit():
        existing = Review.query.filter_by(user_id=current_user.id, property_id=property_id).first()
        if existing:
            flash("You have already reviewed this property.", "warning")
        else:
            review = Review(
                content=form.content.data,
                rating=form.rating.data,
                user_id=current_user.id,
                property_id=property_id,
            )
            db.session.add(review)
            db.session.commit()
            flash("Your review has been added!", "success")
    return redirect(url_for("property_detail", property_id=property_id))


@app.route("/property/<int:property_id>/favorite", methods=["POST"])
@login_required
def toggle_favorite(property_id):
    prop = Property.query.get_or_404(property_id)
    fav = Favorite.query.filter_by(user_id=current_user.id, property_id=property_id).first()
    if fav:
        db.session.delete(fav)
        db.session.commit()
        flash("Property removed from favorites!", "success")
    else:
        fav = Favorite(user_id=current_user.id, property_id=property_id)
        db.session.add(fav)
        db.session.commit()
        flash("Property added to favorites!", "success")
    return redirect(url_for("property_detail", property_id=property_id))


@app.route("/customer/favorites")
@login_required
def customer_favorites():
    if current_user.role != "customer":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))
    favorites = Favorite.query.filter_by(user_id=current_user.id).all()
    return render_template("customer/favorites.html", favorites=favorites)


# ======================================================
# PROFILE & ACCOUNT
# ======================================================

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = EditProfileForm(obj=current_user)

    if form.validate_on_submit():
        current_user.name = form.name.data
        current_user.username = form.username.data
        current_user.bio = form.bio.data
        current_user.email = form.email.data
        current_user.phone = form.phone.data
        current_user.dob = form.dob.data
        current_user.location = form.location.data
        current_user.timezone = form.timezone.data
        current_user.verified = form.verified.data

        pic_file = form.profile_pic.data
        if pic_file and hasattr(pic_file, "filename") and pic_file.filename:
            filename = secure_filename(pic_file.filename)
            filename = f"{uuid.uuid4().hex}_{filename}"
            pic_dir = os.path.join(app.static_folder, "profile_pics")
            os.makedirs(pic_dir, exist_ok=True)
            pic_path = os.path.join(pic_dir, filename)
            pic_file.save(pic_path)
            if current_user.profile_pic and current_user.profile_pic != "default.jpg":
                old_pic_path = os.path.join(pic_dir, current_user.profile_pic)
                if os.path.exists(old_pic_path):
                    os.remove(old_pic_path)
            current_user.profile_pic = filename

        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    profile_data = {
        "profile_pic": current_user.profile_pic,
        "name": current_user.name,
        "username": current_user.username,
        "bio": current_user.bio,
        "email": current_user.email,
        "phone": current_user.phone,
        "dob": str(current_user.dob) if current_user.dob else "",
        "location": current_user.location,
        "timezone": current_user.timezone,
        "verified": current_user.verified,
        "role": current_user.role,
        "member_since": str(current_user.member_since) if current_user.member_since else "",
        "two_factor_enabled": getattr(current_user, "two_factor_enabled", False),
    }

    if current_user.role == "owner":
        properties = Property.query.filter_by(owner_id=current_user.id).all()
        property_ids = [prop.id for prop in properties]
        bookings = Booking.query.filter(Booking.property_id.in_(property_ids)).order_by(Booking.created_at.desc()).all()
        reviews = Review.query.filter(Review.property_id.in_(property_ids)).all()
        earnings_total = sum((b.property.price or 0) for b in bookings if b.status == "confirmed")
        avg_rating = round(sum(r.rating for r in reviews) / len(reviews), 1) if reviews else 0
        active_listings = sum(1 for p in properties if p.available)
        pending_bookings = sum(1 for b in bookings if b.status == "pending")
        confirmed_bookings = sum(1 for b in bookings if b.status == "confirmed")
        from datetime import datetime
        now = datetime.utcnow()
        monthly_earnings = sum((b.property.price or 0) for b in bookings if b.status == "confirmed" and b.created_at.year == now.year and b.created_at.month == now.month)
        favorites_count = Favorite.query.join(Property, Favorite.property_id == Property.id).filter(Property.owner_id == current_user.id).count()

        return render_template(
            "owner/owner-profile.html",
            form=form,
            profile_data=profile_data,
            earnings_total=earnings_total,
            avg_rating=avg_rating,
            active_listings=active_listings,
            pending_bookings=pending_bookings,
            confirmed_bookings=confirmed_bookings,
            monthly_earnings=monthly_earnings,
            favorites_count=favorites_count,
        )
    else:
        return render_template("customer/profile.html", form=form, profile_data=profile_data)


@app.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if current_user.check_password(form.current_password.data):
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash("Your password has been updated!", "success")
            return redirect(url_for("profile"))
        else:
            flash("Current password is incorrect.", "danger")
    return render_template("settings.html", title="Change Password", form=form)


@app.route("/reset_password", methods=["GET", "POST"])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
        flash("An email has been sent with instructions to reset your password.", "info")
        return redirect(url_for("login"))
    return render_template("login.html", title="Reset Password", form=form)


@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    user = User.verify_reset_token(token)
    if user is None:
        flash("That is an invalid or expired token", "warning")
        return redirect(url_for("reset_request"))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash("Your password has been updated! You are now able to log in", "success")
        return redirect(url_for("login"))
    return render_template("login.html", title="Reset Password", form=form)


# ======================================================
# ADMIN: USERS / PROPERTIES
# ======================================================

@app.route("/admin/users")
@login_required
def admin_users():
    if current_user.role != "admin":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))
    users = User.query.all()
    return render_template("customer/dashboard.html", users=users)


@app.route("/admin/user/delete/<int:user_id>", methods=["POST"])
@login_required
def admin_delete_user(user_id):
    if current_user.role != "admin":
        flash("You are not authorized to perform this action", "danger")
        return redirect(url_for("dashboard"))
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("You cannot delete your own account", "danger")
        return redirect(url_for("admin_users"))
    Property.query.filter_by(owner_id=user.id).delete()
    Booking.query.filter_by(customer_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash("User deleted successfully", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/properties")
@login_required
def admin_properties():
    if current_user.role != "admin":
        flash("You are not authorized to access this page", "danger")
        return redirect(url_for("dashboard"))
    properties = Property.query.all()
    return render_template("my-properties.html", properties=properties)


@app.route("/admin/property/delete/<int:property_id>", methods=["POST"])
@login_required
def admin_delete_property(property_id):
    if current_user.role != "admin":
        flash("You are not authorized to perform this action", "danger")
        return redirect(url_for("dashboard"))
    prop = Property.query.get_or_404(property_id)
    Booking.query.filter_by(property_id=prop.id).delete()
    db.session.delete(prop)
    db.session.commit()
    flash("Property deleted successfully", "success")
    return redirect(url_for("admin_properties"))


# ======================================================
# MISC / PLACEHOLDER PAGES
# ======================================================

@app.route("/available_properties")
@login_required
def available_properties():
    properties = Property.query.all()
    return render_template("listing.html", properties=properties)


@app.route("/bookings")
@login_required
def bookings():
    return render_template("bookings.html")


@app.route("/favorites")
@login_required
def favorites():
    return render_template("favorites.html")


@app.route("/reviews")
@login_required
def reviews_page():
    return render_template("listing-details.html")


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    form = EditProfileForm(obj=current_user)
    # Pre-populate split name fields from existing `name`
    try:
        parts = (current_user.name or "").split()
        if not form.first_name.data:
            form.first_name.data = parts[0] if parts else ""
        if not form.middle_name.data:
            form.middle_name.data = parts[1] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "")
        if not form.last_name.data:
            form.last_name.data = parts[-1] if len(parts) > 1 else ""
    except Exception:
        pass
    if form.validate_on_submit():
        # Combine split fields into `name` (fallback to existing `name` field)
        fn = (form.first_name.data or "").strip()
        mn = (form.middle_name.data or "").strip()
        ln = (form.last_name.data or "").strip()
        combined_name = " ".join([p for p in [fn, mn, ln] if p]).strip()
        current_user.name = combined_name or (form.name.data or current_user.name)
        # Auto-generate username if empty
        if (not form.username.data) or (form.username.data.strip()==""):
            base = (fn or current_user.name or "user").lower().replace(" ", "")
            short = base[:12]
            import random
            form.username.data = f"{short}{random.randint(100,999)}"
        current_user.username = form.username.data
        current_user.bio = form.bio.data
        current_user.email = form.email.data
        current_user.phone = form.phone.data
        current_user.dob = form.dob.data
        current_user.location = form.location.data
        current_user.timezone = form.timezone.data
        current_user.verified = form.verified.data

        pic_file = form.profile_pic.data
        if pic_file and hasattr(pic_file, "filename") and pic_file.filename:
            filename = secure_filename(pic_file.filename)
            filename = f"{uuid.uuid4().hex}_{filename}"
            pic_dir = os.path.join(app.root_path, "static", "profile_pics")
            os.makedirs(pic_dir, exist_ok=True)
            pic_path = os.path.join(pic_dir, filename)
            pic_file.save(pic_path)
            if current_user.profile_pic and current_user.profile_pic != "default.jpg":
                old_pic_path = os.path.join(pic_dir, current_user.profile_pic)
                if os.path.exists(old_pic_path):
                    os.remove(old_pic_path)
            current_user.profile_pic = filename

        db.session.commit()
        flash("Account settings updated!", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", form=form)


# ======================================================
# DB INIT (LOCAL DEV) + ENTRYPOINT
# ======================================================

def initialize_database():
    with app.app_context():
        db.create_all()
        admin_email = app.config.get("ADMIN_EMAIL", "admin@example.com")
        admin_user = User.query.filter_by(email=admin_email).first()
        if not admin_user:
            admin = User(
                name="Admin",
                email=admin_email,
                role="admin",
                phone="0000000000",
            )
            admin.set_password(app.config.get("ADMIN_PASSWORD", "admin123"))
            db.session.add(admin)
            db.session.commit()


if __name__ == "__main__":
    initialize_database()
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
# ======================================================
# SAVE PREDICTION RESULTS
# ======================================================

@app.route("/save_prediction", methods=["POST"])
def save_prediction():
    if not current_user.is_authenticated:
        return jsonify({"error": "Login required"}), 401
    payload = request.get_json() or {}
    try:
        rec = PredictionResult(
            user_id=current_user.id,
            predicted_rent=float(payload.get("predicted_rent", 0) or 0),
            city=payload.get("city") or None,
            area_locality=payload.get("area_locality") or None,
            bhk=int(payload.get("bhk", 0) or 0) or None,
            bathroom=int(payload.get("bathroom", 0) or 0) or None,
            size=float(payload.get("size", 0) or 0) or None,
            furnishing_status=payload.get("furnishing_status") or None,
            tenant_preferred=payload.get("tenant_preferred") or None,
            area_type=payload.get("area_type") or None,
            point_of_contact=payload.get("point_of_contact") or None,
            notes=payload.get("notes") or None,
        )
        db.session.add(rec)
        db.session.commit()
        return jsonify({"ok": True, "id": rec.id})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Save prediction error: {e}")
        return jsonify({"error": "Could not save"}), 500

@app.route("/save_analysis", methods=["POST"])
def save_analysis():
    # Reuse save_prediction, but mark notes field
    payload = request.get_json() or {}
    payload["notes"] = (payload.get("notes") or "Detailed analysis saved")
    return save_prediction()
