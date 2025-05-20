import os, base64, json, re
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from image_tool import extract_image_attributes

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Folder containing your images; update this path as needed
default_folder = "/Users/salmolichandra/Desktop/Agent_pre/images"
# Output files
json_output_file = os.path.join(default_folder, "response.json")
txt_output_file = os.path.join(default_folder, "response.txt")

# LLM for summaries
summarizer = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# Supported image extensions
exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')


def clean_json(raw: str) -> str:
    """Strip code fences and wrappers from raw model output to get pure JSON."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"```$", "", s).strip()
    return s


def review_image_attributes(path: str, attrs_json: str) -> str:
    # Read & base64-encode image
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    # Ask second agent to review & correct the JSON
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert image analyst. A first agent extracted the following JSON attributes from the image. "
                "Review these attributes against the image, correct any mistakes, add missing details, and respond with the full corrected JSON object only."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": f"Extracted attributes: {attrs_json}"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def process_folder(folder_path: str,
                   json_output_file: str = json_output_file,
                   txt_output_file: str = txt_output_file):
    records = []
    summaries = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(exts):
            continue
        path = os.path.join(folder_path, filename)
        print(f"Processing image: {path}")

        # 1. First agent: extract attributes
        raw = extract_image_attributes(path)
        json_str = clean_json(raw)
        try:
            attrs = json.loads(json_str)
        except json.JSONDecodeError:
            attrs = json_str

        # 2. Second agent: review & correct
        review_raw = review_image_attributes(path, json.dumps(attrs))
        review_json_str = clean_json(review_raw)
        try:
            reviewed = json.loads(review_json_str)
        except json.JSONDecodeError:
            reviewed = review_json_str

        # 3. Clean up colors list (drop hex codes)
        if isinstance(reviewed, dict) and 'colors' in reviewed and isinstance(reviewed['colors'], list):
            clean_colors = []
            for c in reviewed['colors']:
                if isinstance(c, dict) and 'name' in c:
                    clean_colors.append(c['name'])
                elif isinstance(c, str):
                    clean_colors.append(c)
            reviewed['colors'] = clean_colors

        # 4. Collect record
        records.append({
            "image": path,
            "attributes": reviewed
        })

        # 5. Generate summary
        summary = summarizer.predict(
            f"Summarize these attributes into one sentence, using only color names (no hex codes): {json.dumps(reviewed)}"
        )
        summaries.append({"image": path, "summary": summary})

    # 6. Write outputs
    with open(json_output_file, 'w') as jf:
        json.dump(records, jf, indent=2)
    with open(txt_output_file, 'w') as tf:
        for e in summaries:
            tf.write(f"Image: {e['image']}\n{e['summary']}\n\n")

    print(f"All done. JSON saved to: {json_output_file}")
    print(f"Summaries saved to: {txt_output_file}")


if __name__ == "__main__":
    process_folder(default_folder)
