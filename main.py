"""
Medical Transcription & SOAP Note Generator
Flask web application with Anthropic Claude API
"""

import os
import json
import base64
import re
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
import anthropic

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

UPLOAD_DIR = Path("/app/audio")
OUTPUT_DIR = Path("/app/outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SOAP_SYSTEM_PROMPT = """You are a clinical documentation specialist with expertise in medical transcription.
Convert raw physician dictation into a structured SOAP note.

SOAP definitions:
- S (Subjective): Patient-reported info — symptoms, complaints, pain levels, medications they mention, history.
- O (Objective): Clinician-observed data ONLY — vital signs, physical exam findings, lab results, imaging. Never patient-reported feelings.
- A (Assessment): Clinician's diagnosis or clinical impression.
- P (Plan): Treatment, prescriptions, referrals, follow-up instructions.

CRITICAL BOUNDARY RULE:
  CORRECT Subjective: "Patient reports pain 7/10"
  CORRECT Objective: "Tenderness on palpation noted"
  WRONG: Patient-reported pain ratings in Objective section.

Respond ONLY with valid JSON, no markdown fences, no preamble:
{
  "patient_info": {
    "age": "string or null",
    "sex": "string or null",
    "chief_complaint": "one-line summary"
  },
  "soap_note": {
    "subjective": {
      "chief_complaint": "string",
      "history_of_present_illness": "string",
      "pain_scale": "string or null",
      "current_medications": ["list"],
      "other_subjective": "string or null"
    },
    "objective": {
      "vital_signs": "string or null",
      "physical_exam": "string",
      "diagnostic_results": "string or null"
    },
    "assessment": {
      "diagnosis": "string",
      "clinical_impression": "string"
    },
    "plan": {
      "medications": ["list"],
      "procedures": ["list"],
      "follow_up": "string",
      "patient_education": "string or null"
    }
  },
  "validation": {
    "subjective_objective_boundary_check": "PASS or FAIL",
    "boundary_check_notes": "explanation"
  }
}"""

SUBJECTIVE_MARKERS = [
    "patient reports", "patient states", "patient denies", "patient complains",
    "he reports", "she reports", "he says", "she says", "he denies", "she denies",
    "rates the pain", "rates pain", "out of 10", "/10",
    "patient feels", "patient notes", "patient endorses", "according to the patient",
]


def get_client():
    key = ANTHROPIC_API_KEY or request.headers.get("X-API-Key", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def transcribe_audio(audio_path: str, mime_type: str = "audio/mpeg") -> dict:
    """Transcribe audio using Claude's native audio understanding."""
    client = get_client()

    with open(audio_path, "rb") as f:
        audio_b64 = base64.standard_b64encode(f.read()).decode()

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": audio_b64
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Transcribe this medical dictation audio verbatim, word for word. "
                        "Preserve all medical terminology exactly as spoken. "
                        "Output only the raw transcript — no labels, no commentary, no timestamps."
                    )
                }
            ]
        }]
    )

    transcript = " ".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    return {
        "transcript": transcript,
        "model": "claude-sonnet-4-5",
        "generated_at": datetime.now().isoformat(),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def generate_soap(transcript: str) -> dict:
    """Generate structured SOAP note from transcript using Claude."""
    client = get_client()

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=SOAP_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Convert this medical dictation into a structured SOAP note:\n\n{transcript}"
        }]
    )

    raw = " ".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

    soap_data = json.loads(raw)
    return soap_data


def validate_boundaries(soap_data: dict) -> dict:
    """Programmatic check: ensure no subjective markers appear in Objective section."""
    obj = soap_data.get("soap_note", {}).get("objective", {})
    obj_text = " ".join(
        str(v).lower() for v in obj.values() if v
    )

    issues = [m for m in SUBJECTIVE_MARKERS if m in obj_text]
    programmatic = "PASS" if not issues else "FAIL"
    llm_check = soap_data.get("validation", {}).get(
        "subjective_objective_boundary_check", "NOT_REPORTED"
    )

    return {
        "programmatic_check": programmatic,
        "llm_self_check": llm_check,
        "issues_found": issues,
        "overall_status": "PASS" if programmatic == "PASS" and llm_check == "PASS" else "REVIEW_NEEDED"
    }


def soap_to_markdown(soap_data: dict, transcript: str = "") -> str:
    """Format SOAP JSON as human-readable Markdown."""
    s = soap_data.get("soap_note", {}).get("subjective", {})
    o = soap_data.get("soap_note", {}).get("objective", {})
    a = soap_data.get("soap_note", {}).get("assessment", {})
    p = soap_data.get("soap_note", {}).get("plan", {})
    pi = soap_data.get("patient_info", {})
    val = soap_data.get("validation", {})

    meds = ", ".join(s.get("current_medications", [])) or "None reported"
    plan_meds = ", ".join(p.get("medications", [])) or "None"
    plan_proc = ", ".join(p.get("procedures", [])) or "None"

    lines = [
        "# SOAP Note",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model:** claude-sonnet-4-5 (Claude API)\n",
        "---\n",
        "## Patient Information",
        f"- **Age:** {pi.get('age', 'N/A')}",
        f"- **Sex:** {pi.get('sex', 'N/A')}",
        f"- **Chief Complaint:** {pi.get('chief_complaint', 'N/A')}\n",
        "---\n",
        "## S — Subjective *(Patient-reported)*",
        f"**Chief Complaint:** {s.get('chief_complaint', 'N/A')}  ",
        f"**HPI:** {s.get('history_of_present_illness', 'N/A')}  ",
        f"**Pain Scale:** {s.get('pain_scale', 'N/A')}  ",
        f"**Current Medications:** {meds}  ",
    ]
    if s.get("other_subjective"):
        lines.append(f"**Other:** {s['other_subjective']}")

    lines += [
        "\n---\n",
        "## O — Objective *(Clinician-observed)*",
        f"**Vital Signs:** {o.get('vital_signs', 'Not documented')}  ",
        f"**Physical Exam:** {o.get('physical_exam', 'N/A')}  ",
        f"**Diagnostic Results:** {o.get('diagnostic_results', 'None')}  ",
        "\n---\n",
        "## A — Assessment *(Diagnosis)*",
        f"**Diagnosis:** {a.get('diagnosis', 'N/A')}  ",
        f"**Clinical Impression:** {a.get('clinical_impression', 'N/A')}  ",
        "\n---\n",
        "## P — Plan *(Treatment)*",
        f"**Medications:** {plan_meds}  ",
        f"**Procedures:** {plan_proc}  ",
        f"**Follow-up:** {p.get('follow_up', 'N/A')}  ",
    ]
    if p.get("patient_education"):
        lines.append(f"**Patient Education:** {p['patient_education']}")

    lines += [
        "\n---\n",
        "## ✅ Validation",
        f"**S/O Boundary Check:** {val.get('subjective_objective_boundary_check', 'N/A')}  ",
        f"**Notes:** {val.get('boundary_check_notes', 'N/A')}  ",
    ]

    if transcript:
        lines += ["\n---\n", "## Raw Transcript", f"\n{transcript}"]

    return "\n".join(lines)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "claude-sonnet-4-5"})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Upload audio file and get transcript."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(audio_file.filename).suffix.lower()
    mime_map = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".webm": "audio/webm"
    }
    mime = mime_map.get(ext, "audio/mpeg")

    save_path = UPLOAD_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    audio_file.save(str(save_path))

    try:
        result = transcribe_audio(str(save_path), mime)
        # Save transcript
        ts_file = OUTPUT_DIR / "raw_transcript.txt"
        with open(ts_file, "w") as f:
            f.write(f"Medical Dictation Transcript\n")
            f.write(f"Generated: {result['generated_at']}\n")
            f.write(f"Model: {result['model']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(result["transcript"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/soap", methods=["POST"])
def api_soap():
    """Generate SOAP note from transcript text."""
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "No transcript provided"}), 400

    transcript = data["transcript"].strip()
    if not transcript:
        return jsonify({"error": "Transcript is empty"}), 400

    try:
        soap_data = generate_soap(transcript)
        validation = validate_boundaries(soap_data)

        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": "claude-sonnet-4-5",
                "transcript_word_count": len(transcript.split())
            },
            "raw_transcript": transcript,
            "soap_note": soap_data,
            "validation": validation
        }

        # Save outputs
        json_file = OUTPUT_DIR / "soap_note.json"
        with open(json_file, "w") as f:
            json.dump(output, f, indent=2)

        md_file = OUTPUT_DIR / "soap_note.md"
        with open(md_file, "w") as f:
            f.write(soap_to_markdown(soap_data, transcript))

        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/outputs/<filename>")
def download_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
