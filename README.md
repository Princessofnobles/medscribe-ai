# 🏥 MedScribe AI — Medical Transcription & SOAP Note Generator

> **Technical Assignment** | Medical dictation → verbatim transcript → structured SOAP note, fully automated with AI.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Claude](https://img.shields.io/badge/Claude-claude--sonnet--4--5-orange)](https://anthropic.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Assignment Objective

Build a pipeline that:
1. **Ingests** a medical dictation audio file
2. **Transcribes** it verbatim with high medical-term accuracy
3. **Structures** the transcript into a standardized [SOAP note](https://en.wikipedia.org/wiki/SOAP_note)

**Output formats:** JSON + Markdown  
**Code:** Python (Jupyter Notebook + Flask web app)  
**Deployment:** Docker

---

## 🏗️ Architecture

```
sample_dictation.mp3
        │
        ▼
┌───────────────────┐
│  Claude claude-sonnet-4-5  │  ← Part A: Audio → Transcript
│  (audio input)    │     verbatim, medical terms preserved
└───────────────────┘
        │
        ▼ raw_transcript.txt
        │
┌───────────────────┐
│  Claude claude-sonnet-4-5  │  ← Part B: Transcript → SOAP Note
│  (prompt eng.)    │     structured JSON with boundary check
└───────────────────┘
        │
        ▼
   soap_note.json
   soap_note.md
```

### Why Claude claude-sonnet-4-5 for transcription?

| Factor | Claude claude-sonnet-4-5 | Generic STT (Whisper/Google) |
|--------|--------------|------------------------------|
| Medical terminology | ✅ Excellent | ⚠️ Requires fine-tuning |
| Drug names & dosages | ✅ Accurate | ⚠️ Often mis-transcribes |
| Setup complexity | ✅ One API call | ❌ Model download / API key |
| Privacy | ✅ Encrypted, not stored | Varies |
| Local fallback | Use `--ollama` flag | Whisper (see below) |

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/medscribe-ai.git
cd medscribe-ai

# 2. Configure
cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=sk-ant-...

# 3. Add the audio file
cp /path/to/sample_dictation.mp3 audio/

# 4. Build & run
docker-compose up --build

# 5. Open browser
open http://localhost:8080
```

### Option B — Jupyter Notebook

```bash
# Install dependencies
pip install anthropic python-dotenv jupyter

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run notebook
jupyter notebook medical_transcription_pipeline.ipynb
```

### Option C — Python Script

```bash
pip install anthropic python-dotenv

export ANTHROPIC_API_KEY=sk-ant-...

python app/main.py   # starts Flask server on :8080
# or run directly:
# python -c "from app.main import *; ..."
```

---

## 📁 Repository Structure

```
medscribe-ai/
├── 📓 medical_transcription_pipeline.ipynb   # Main notebook (assignment deliverable)
├── 🐍 app/
│   ├── main.py                               # Flask web application
│   └── templates/
│       └── index.html                        # Web UI
├── 🎵 audio/
│   └── sample_dictation.mp3                  # Provided audio file
├── 📂 outputs/                               # Generated outputs (git-ignored)
│   ├── raw_transcript.txt
│   ├── soap_note.json
│   └── soap_note.md
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── 📋 requirements.txt
├── ⚙️  .env.example
└── 📖 README.md
```

---

## 🌐 Live Demo

**Deployed on Render:** [https://medscribe-ai.onrender.com](https://medscribe-ai.onrender.com)

> To deploy your own instance, see [Deploy to Render](#deploy-to-render) below.

---

## 📤 Sample Output

### Raw Transcript (`outputs/raw_transcript.txt`)

```
Patient is a 52-year-old female presenting today with complaints of persistent
shortness of breath and chest tightness that began approximately three days ago.
She rates her dyspnea as a 6 out of 10 at rest and worsens to 9 out of 10 with
minimal exertion...
```

### SOAP Note (`outputs/soap_note.json`)

```json
{
  "patient_info": {
    "age": "52 years old",
    "sex": "Female",
    "chief_complaint": "Persistent shortness of breath and chest tightness x3 days"
  },
  "soap_note": {
    "subjective": {
      "chief_complaint": "Shortness of breath and chest tightness for 3 days",
      "history_of_present_illness": "52yo female with hx of HTN and T2DM presenting with 3-day progressive dyspnea...",
      "pain_scale": "Dyspnea 6/10 rest, 9/10 with exertion",
      "current_medications": ["Lisinopril 10mg daily", "Metformin 500mg BID"],
      "other_subjective": "Penicillin allergy. Mild dry cough, denies fever/hemoptysis."
    },
    "objective": {
      "vital_signs": "BP 148/92 | HR 98 | RR 22 | SpO2 91% RA | Temp 37.1°C",
      "physical_exam": "Bilateral basilar crackles. Bilateral LE pitting edema to mid-calf.",
      "diagnostic_results": "CXR: Cardiomegaly + pulmonary edema. BNP: 820 pg/mL (elevated)."
    },
    "assessment": {
      "diagnosis": "Acute Decompensated Heart Failure (ADHF)",
      "clinical_impression": "ADHF likely precipitated by medication non-compliance..."
    },
    "plan": {
      "medications": ["IV Furosemide 40mg stat"],
      "procedures": ["Admit cardiac unit", "Cardiology consult", "Daily weights"],
      "follow_up": "PCP within 1 week of discharge",
      "patient_education": "Sodium restriction, medication adherence, warning signs"
    }
  },
  "validation": {
    "subjective_objective_boundary_check": "PASS",
    "boundary_check_notes": "All patient-reported info in Subjective. All clinical observations in Objective."
  }
}
```

---

## ✅ Validation Logic

Two-layer boundary check ensures no subjective (patient-reported) content bleeds into the Objective section:

1. **LLM self-check** — Claude evaluates its own output and reports PASS/FAIL
2. **Programmatic check** — Rule-based scan of the Objective section for patient-report markers:
   - "patient reports", "he/she says", "rates pain", "/10", "patient feels", etc.

Both must PASS for `overall_status: PASS`.

---

## 🚢 Deploy to Render (Free)

1. Fork this repo on GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --timeout 300 app.main:app`
   - **Environment Variable:** `ANTHROPIC_API_KEY` = your key
5. Deploy → get your live URL

---

## 🐳 Docker Details

```bash
# Build image
docker build -t medscribe-ai .

# Run with API key
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... medscribe-ai

# Or with docker-compose (reads from .env)
docker-compose up --build

# Stop
docker-compose down
```

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| POST | `/api/transcribe` | Upload audio → transcript |
| POST | `/api/soap` | Transcript text → SOAP note |
| GET | `/api/outputs/<file>` | Download output files |

**Transcribe request:**
```bash
curl -X POST http://localhost:8080/api/transcribe \
  -H "X-API-Key: sk-ant-..." \
  -F "audio=@audio/sample_dictation.mp3"
```

**SOAP generation request:**
```bash
curl -X POST http://localhost:8080/api/soap \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-ant-..." \
  -d '{"transcript": "Patient is a 45yo male..."}'
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| `PORT` | `8080` | Server port |

---

## 📦 Dependencies

```
flask==3.1.0          # Web framework
anthropic==0.40.0     # Claude API client
python-dotenv==1.0.1  # .env file support
gunicorn==23.0.0      # Production WSGI server
```

---

## 💡 Scaling Considerations

- **High volume:** Use async processing with Celery + Redis for concurrent audio jobs
- **Fully offline:** Replace Claude API with `ollama run llama3` + local Whisper for STT
- **EHR integration:** Output JSON maps directly to HL7 FHIR `DocumentReference` resource
- **GPU acceleration:** Add Faster-Whisper with CUDA for on-premise STT at scale

---

## 👩‍💻 Author

**Sulthana Parveen** | ParvinAI Agency  
Digital Marketing & AI Automation Specialist · Kerala, India

---

*Built for technical assessment — Medical Transcription & SOAP Note Generation Pipeline*
