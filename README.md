# PCOS Prediction Web App

A production-ready, mobile-first, and fully responsive PCOS (Polycystic Ovary Syndrome) Prediction Web Application built with Python Flask and Machine Learning. The application features full PWA capabilities, allowing installation on mobile and desktop home screens.

---

## Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the ML Model:**
   ```bash
   python train_model.py
   ```
   *This loads the dataset `data/PCOS_data.csv`, trains the VotingClassifier soft-voting ensemble model, and exports the serialized model assets into the `model/` directory.*

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your_flask_secret_key
   GEMINI_API_KEY=your_google_gemini_api_key
   MAIL_USERNAME=your_gmail_address@gmail.com
   MAIL_PASSWORD=your_gmail_app_password
   ```

4. **Launch the Flask Application:**
   ```bash
   python app.py
   ```

5. **Access the App:**
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## Deploy to Render.com (Free)

1. Push your code to a public or private GitHub repository.
2. Sign up or log in to the [Render Dashboard](https://dashboard.render.com).
3. Click **New** &rarr; **Web Service**.
4. Connect your GitHub repository.
5. Set the following parameters during creation:
   - **Environment:** `Python 3`
   - **Region:** Choose the closest region.
   - **Branch:** `main` (or your active development branch).
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 2 --timeout 120`
6. Go to the **Environment** tab in your Web Service dashboard and define these environment variables:
   - `SECRET_KEY` = *any_random_string_key*
   - `DATABASE_URL` = `sqlite:///database/pcos_app.db`
   - `GEMINI_API_KEY` = *your_google_gemini_api_key*
   - `MAIL_USERNAME` = *your_gmail_address@gmail.com*
   - `MAIL_PASSWORD` = *your_gmail_app_password*
7. Click **Deploy Web Service**. Your app will be live at `https://yourappname.onrender.com`.

---

## Deploy to Railway.app (Free)

1. Install the Railway CLI globally:
   ```bash
   npm install -g @railway/cli
   ```
2. Log in using your credentials:
   ```bash
   railway login
   ```
3. Initialize the project in your local repository folder:
   ```bash
   railway init
   ```
4. Deploy the application:
   ```bash
   railway up
   ```
5. Configure your environmental values (e.g. `GEMINI_API_KEY`, `SECRET_KEY`, `MAIL_USERNAME`, `MAIL_PASSWORD`) under the **Variables** settings tab in the Railway Dashboard.
6. Generate a domain name under **Settings** to make the site live at `https://yourapp.railway.app`.

---

## Deploy to PythonAnywhere (Free)

1. Sign up for a free account at [PythonAnywhere](https://www.pythonanywhere.com).
2. Go to the **Files** tab and upload all files from your project directory.
3. Open a **Bash Console** in PythonAnywhere and run:
   ```bash
   pip install --user -r requirements.txt
   python train_model.py
   ```
4. Navigate to the **Web** tab &rarr; Click **Add a new web app**.
5. Select **Flask** framework, and select **Python 3.11** as the version.
6. In the configuration settings, verify:
   - **Source code path:** set to your uploaded project directory (e.g., `/home/username/pcos_app`).
   - **Working directory:** set to the same folder.
7. Open the **WSGI configuration file** (linked on the Web tab configuration section) and ensure it loads your Flask app instance (`from app import app`).
8. Define any required environment variables under Web configuration or inside the WSGI file.
9. Click **Reload** at the top of the Web tab. Your app will be live at `https://username.pythonanywhere.com`.

---

## Device Compatibility & PWA Support

The application is built with mobile-first CSS grids and is optimized for the following configurations:
- **Mobile Devices:** (e.g., iPhone SE, 13, 14, 15, Samsung Galaxy) &mdash; elements display in a single-column layout, with minimum tap target bounds of 44x44px. Font sizes on inputs are $\ge$ 16px to prevent iOS auto-zooming.
- **Tablets:** (e.g., iPad Mini, iPad Pro) &mdash; forms display in a 2-column layout.
- **Desktops:** (e.g., standard laptops, monitors) &mdash; forms display in a balanced 3-column layout.
- **Browser Matrices:** Fully compatible with Google Chrome, Apple Safari, Mozilla Firefox, Microsoft Edge, and Samsung Internet.
- **Installable PWA:** Can be pinned to mobile or desktop home screens via browser options (e.g., "Add to Home Screen" or "Install App"), providing instant accessibility and basic offline caching functions.
