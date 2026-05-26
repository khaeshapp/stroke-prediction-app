# AI Stroke Risk Prediction System

A full-stack healthcare application for predicting stroke risk using machine learning.

## Architecture

- **Frontend**: React + TypeScript + Tailwind CSS (Vite)
- **Backend Options**:
  - Flask (Python) - Local development
  - Supabase Edge Functions (Deno/TypeScript) - Cloud deployment

---

## Quick Start

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will run on `http://localhost:5173`

### Backend Setup - Flask (Recommended for Local Development)

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Flask server**:
   ```bash
   python app.py
   ```

   The API will run on `http://localhost:5000`

3. **Verify API is running**:
   ```bash
   curl http://localhost:5000/health
   ```

### Backend Setup - Supabase (Cloud Deployment)

The Supabase Edge Function is already deployed. To switch between backends:

1. Update `.env` file:
   ```env
   VITE_API_MODE=supabase
   ```

2. Restart the frontend development server

---

## API Endpoints

### Flask Backend

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Predict stroke risk |
| `/history` | GET | Get prediction history |
| `/model-info` | GET | Get ML model information |

### Example API Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patientData": {
      "age": 65,
      "gender": "Male",
      "hypertension": "Yes",
      "heartDisease": "No",
      "everMarried": "Yes",
      "residenceType": "Urban",
      "avgGlucoseLevel": 180,
      "bmi": 28.5,
      "workType": "Private",
      "smokingStatus": "NeverSmoked"
    }
  }'
```

### Example Response

```json
{
  "success": true,
  "prediction": {
    "risk_level": "medium",
    "risk_percentage": 48,
    "confidence": 85,
    "risk_factors": [
      "Age increases stroke risk after 55",
      "Hypertension is a major stroke risk factor",
      "Elevated glucose affects vascular health",
      "Consider lifestyle modifications"
    ]
  },
  "timestamp": "2026-05-26T10:30:00"
}
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Supabase (for cloud deployment)
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Backend Mode: 'flask' or 'supabase'
VITE_API_MODE=flask
VITE_FLASK_API_URL=http://localhost:5000
```

### Switching Backends

**Use Flask (Local)**:
```env
VITE_API_MODE=flask
VITE_FLASK_API_URL=http://localhost:5000
```

**Use Supabase (Cloud)**:
```env
VITE_API_MODE=supabase
```

---

## Project Structure

```
project/
├── app.py                      # Flask backend
├── requirements.txt            # Python dependencies
├── src/
│   ├── App.tsx                # Main React component
│   ├── lib/
│   │   └── supabase.ts        # Supabase client
│   └── services/
│       └── predictionService.ts  # API integration
├── supabase/
│   └── functions/
│       └── stroke-prediction/
│           └── index.ts       # Supabase Edge Function
└── .env                       # Environment configuration
```

---

## ML Prediction Algorithm

The prediction model uses a weighted scoring system:

| Factor | Weight |
|--------|--------|
| Age 65+ | 20 points |
| Hypertension | 25 points |
| Heart Disease | 25 points |
| Current Smoker | 15 points |
| High Glucose (>200) | 20 points |
| Obesity (BMI >30) | 10-15 points |

**Risk Levels**:
- **Low**: 0-29%
- **Medium**: 30-59%
- **High**: 60%+

---

## Development

### Build for Production

```bash
npm run build
```

### Run Type Checks

```bash
npm run typecheck
```

### Run Linter

```bash
npm run lint
```

---

## Deployment

### Frontend (Vercel, Netlify, etc.)

1. Build the project:
   ```bash
   npm run build
   ```

2. Deploy the `dist/` folder

### Backend

- **Flask**: Deploy to Heroku, Render, Railway, or any Python hosting
- **Supabase**: Already deployed! Just use `VITE_API_MODE=supabase`

---

## Technology Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, Vite
- **Backend**: Flask (Python), Supabase Edge Functions
- **Database**: Supabase PostgreSQL
- **Icons**: Lucide React
- **ML**: Weighted scoring algorithm

---

## License

MIT License - For educational and demonstration purposes.
