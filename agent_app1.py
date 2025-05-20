import os
import json
import re
from langchain_community.chat_models import ChatOpenAI
from image_tool import extract_image_attributes

# Folder containing your images; update this path as needed
default_folder = "/Users/salmolichandra/Desktop/Agent_pre/images"
# Output files
json_output_file = os.path.join(default_folder, "response.json")
txt_output_file = os.path.join(default_folder, "response.txt")

# Initialize the LLM for summarization
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
# Supported image extensions
exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')


def clean_json(raw: str) -> str:
    """Strip code fences and wrappers from raw model output to get pure JSON."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"```$", "", s).strip()
    return s


def process_folder(folder_path: str):
    records = []
    summaries = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(exts):
            continue
        path = os.path.join(folder_path, filename)
        print(f"Processing image: {path}")

        # 1. Extract and parse JSON attributes
        raw = extract_image_attributes(path)
        json_str = clean_json(raw)
        try:
            attrs = json.loads(json_str)
        except json.JSONDecodeError:
            attrs = json_str

        # 2. Drop any hex codes under colors
        if isinstance(attrs, dict) and 'colors' in attrs and isinstance(attrs['colors'], list):
            cleaned_colors = []
            for c in attrs['colors']:
                if isinstance(c, dict) and 'name' in c:
                    cleaned_colors.append(c['name'])
                elif isinstance(c, str):
                    cleaned_colors.append(c)
            attrs['colors'] = cleaned_colors

        records.append({
            "image": path,
            "attributes": attrs
        })

        # 3. Generate summary
        summary = llm.predict(
            f"Summarize these attributes into one sentence, using only color names (no hex codes): {json.dumps(attrs)}"
        )
        summaries.append({"image": path, "summary": summary})

    # 4. Write full JSON array with pretty formatting
    with open(json_output_file, 'w') as jf:
        json.dump(records, jf, indent=2)

    # 5. Write summaries.txt
    with open(txt_output_file, 'w') as tf:
        for entry in summaries:
            tf.write(f"Image: {entry['image']}\n{entry['summary']}\n\n")

    print(f"All done. JSON saved to: {json_output_file}")
    print(f"Summaries saved to: {txt_output_file}")


if __name__ == "__main__":
    process_folder(default_folder)
