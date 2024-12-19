import modal
from modal import App, web_endpoint

# Create a Modal image with FastAPI installed
image = (
    modal.Image.debian_slim()
    .pip_install("fastapi[standard]")
)

# Create the Modal app with our image
app = App("web-endpoint-example", image=image)

# Create a web endpoint
@app.function()
@web_endpoint(method="POST")
def generate(request):
    # Get the prompt from the request body
    data = request.json
    prompt = data.get("prompt", "")
    
    # Process the prompt (example response)
    response = {
        "message": f"Received prompt: {prompt}",
        "status": "success"
    }
    
    return response

# You can also create other regular Modal functions
@app.function()
def helper_function(text: str):
    return text.upper()