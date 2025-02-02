# Models for each processor
IMAGE_PROCESSOR_MODELS = {
    "chatgpt":  "gpt-4o-mini",
    "gemini": "gemini-1.5-flash-8b",
    "claude": "claude-3-opus-20240229",
    "groq": "llama-3.2-90b-vision-preview"
}

CODE_PROCESSOR_MODELS = {
    "chatgpt": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash-8b",
    "claude": "claude-3-opus-20240229",
    "groq": "llama-3.3-70b-versatile"
}

DOCUMENT_PROCESSOR_MODELS = {    
    "chatgpt": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash-8b",
    "claude": "claude-3-opus-20240229",
    "groq": "llama-3.3-70b-versatile"
}

# LLM Settings for each processor  
TEMPERATURES = {
    "image": 0.0,
    "code": 0.0,
    "document": 0.0}