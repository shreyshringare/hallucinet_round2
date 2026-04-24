---
title: HalluciNet Adversarial
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---HalluciNet Adversarial — Round 2

Multi-agent, self-improving hallucination detection for OpenEnv.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-2.0-success)]()
[![Theme 1: Multi-Agent](https://img.shields.io/badge/Theme-Multi--Agent-blue)]()
[![Theme 4: Self-Improvement](https://img.shields.io/badge/Theme-Self--Improvement-purple)]()

## Links
- **HF Space**: https://rushikeshbathe096-hallucinet-adversarial.hf.space
- **GitHub**: https://github.com/shreyshringare/hallucinet_round2
- **Round 1 Environment**: https://rushikeshbathe096-hallucination-detector.hf.space

## Overview
This project has two agents:
- **Generator**: creates subtle factual errors to fool a detector.
- **Detector**: finds hallucinations using reference-grounded checks.

Difficulty advances with an adaptive curriculum across `easy`, `medium`, `hard`, and `expert`.

## API Endpoints
- `GET /health`
- `GET /adversarial/info`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /generator/reset`
- `POST /generator/step`
- `GET /generator/state`
- `GET /` and `GET /demo` (web UI)

## Action Formats

### Detector (`/step`)
```json
{
  "action": {
    "has_hallucination": true,
    "hallucinated_claim": "completed in 1902",
    "correct_fact": "completed in 1889",
    "confidence": 0.95
  }
}
```

### Generator (`/generator/step`)
```json
{
  "action": {
    "generated_response": "The Eiffel Tower was completed in 1902...",
    "error_type": "year_swap",
    "confidence": 0.80
  }
}
```

## Scoring
Detector grading uses weighted components:
- Detection: `0.50`
- Phrase identification: `0.30`
- Correct fact: `0.20`
- Calibration: `±0.10`

Each detector step now includes `reward_breakdown` in observation metadata:
- `detection`, `phrase`, `fact`, `calibration`, `final`

Generator scoring tracks:
- Fluency, subtlety, error-type complexity, calibration

## Reliability Guards
- Step timeout protection in both detector and generator environments (default 30s).
- Timeout returns a terminal observation with `metadata.timeout = true`.
- Curriculum promotion requires a full history window before promotion.

## Local Run
From repo root:

```bash
# install dependencies in your environment first, then:
python -m server.app
```

or:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

The Python script entrypoint is:

```toml
[project.scripts]
server = "server.app:main"
```

## Quick Test (Hosted)
```bash
curl https://rushikeshbathe096-hallucinet-adversarial.hf.space/health
curl https://rushikeshbathe096-hallucinet-adversarial.hf.space/adversarial/info
```

## Deploy (OpenEnv)
```bash
openenv push --repo-id <your-namespace>/<your-space-name>
```

## Team
Abeer Nikhil Sane | Shreyas Shringare | Rushikesh Bathe  
Meta PyTorch OpenEnv Hackathon × Scaler 2026

