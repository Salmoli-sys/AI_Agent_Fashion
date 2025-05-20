# image_tool.py
import os, base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_image_attributes(path: str) -> str:
    # 1) Read & base64-encode
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    data_url = f"data:image/jpeg;base64,{b64}"

    # 2) Pass as an image_url block
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert image analyst.You need to give this to a designer after analysing such that if you give this , he will make the exact same cloth.Neck type and each and every small details should be perfect ."
                    "Given an image, output a JSON object with keys "
                    "like 'objects', 'colors', 'style', 'text', etc."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image; reply with JSON."},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ],
            },
        ],
        max_tokens=1024,
    )

    return resp.choices[0].message.content
