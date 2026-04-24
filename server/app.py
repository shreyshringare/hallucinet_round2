import os
import sys

# Add project root to path so all imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from server.environment import HallucinationEnvironment
from server.generator_environment import GeneratorEnvironment
from models import HallucinationAction, GeneratorAction

app = FastAPI(title="HalluciNet Adversarial - Round 2")

detector_env = HallucinationEnvironment()
generator_env = GeneratorEnvironment()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class DetectorStepRequest(BaseModel):
    action: HallucinationAction

class GeneratorStepRequest(BaseModel):
    action: GeneratorAction

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "adversarial", "version": "2.0"}

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    obs = detector_env.reset(task_id=body.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/step")
def step(body: DetectorStepRequest):
    obs = detector_env.step(body.action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

@app.get("/state")
def state():
    return detector_env.state().model_dump()

@app.post("/generator/reset")
def generator_reset(body: ResetRequest = ResetRequest()):
    obs = generator_env.reset(task_id=body.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/generator/step")
def generator_step(body: GeneratorStepRequest):
    obs = generator_env.step(body.action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

@app.get("/generator/state")
def generator_state():
    return generator_env.state().model_dump()

@app.get("/adversarial/info")
def adversarial_info():
    return {
        "description": "HalluciNet Adversarial - Multi-Agent Self-Play",
        "generator": {
            "endpoint": "/generator/reset and /generator/step",
            "action_space": {
                "generated_response": "string",
                "error_type": "string",
                "confidence": "float strictly between 0 and 1"
            }
        },
        "detector": {
            "endpoint": "/reset and /step",
            "action_space": {
                "has_hallucination": "bool",
                "hallucinated_claim": "string or null",
                "correct_fact": "string or null",
                "confidence": "float strictly between 0 and 1"
            }
        },
        "tasks": ["easy", "medium", "hard", "expert"],
        "themes": ["Theme 1: Multi-Agent", "Theme 4: Self-Improvement"]
    }

@app.get("/leaderboard")
def leaderboard():
    return {"leaderboard": [], "description": "Error types ranked by fool rate"}

@app.get("/stats")
def stats():
    return {"total_detector_episodes": 0, "total_generator_episodes": 0}

try:
    from sample_generator import generate_batch
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False

@app.get("/generate")
def generate_samples(n: int = 10):
    if not GENERATOR_AVAILABLE:
        return {"error": "Sample generator not available", "samples": []}
    samples = generate_batch(n=n, clean_ratio=0.2)
    return {"samples": samples, "count": len(samples), "generated": True}

@app.get("/")
@app.get("/demo")
def demo_ui():
    return HTMLResponse("<html><body><h1>HalluciNet Adversarial</h1><p>Use /reset to start</p></body></html>")


@app.get("/metadata")
def metadata():
    return {
        "name": "hallucinet-adversarial",
        "description": "Adversarial self-improving hallucination detection. Generator vs Detector multi-agent RL. Theme 1 + Theme 4.",
        "version": "2.0.0",
        "author": "team-tle"
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "has_hallucination": "bool",
            "hallucinated_claim": "string or null",
            "correct_fact": "string or null",
            "confidence": "float strictly between 0 and 1"
        },
        "observation": {
            "reference_document": "string",
            "llm_response": "string",
            "feedback": "string",
            "score": "float",
            "reward": "float",
            "done": "bool"
        },
        "state": {
            "episode_id": "string",
            "task_id": "string",
            "steps_taken": "int",
            "is_done": "bool"
        }
    }


@app.post("/mcp")
async def mcp_endpoint(request: dict = None):
    """MCP JSON-RPC endpoint for OpenEnv compatibility."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "name": "hallucinet-adversarial",
            "version": "2.0.0",
            "description": "Adversarial hallucination detection RL environment"
        }
    }

def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False
    )

if __name__ == "__main__":
    main()
