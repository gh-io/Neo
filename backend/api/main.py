from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Prompt(BaseModel):
    text: str

@app.post("/infer")
def infer(payload: Prompt):
    output = f"Neomind processed: {payload.text}"
    return {"output": output}
