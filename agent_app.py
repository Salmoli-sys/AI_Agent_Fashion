# # agent_app.py
# from langchain.agents import initialize_agent, Tool
# from langchain_community.chat_models import ChatOpenAI

# from image_tool import extract_image_attributes

# tools = [
#     Tool(
#         name="ImageAttributeExtractor",
#         func=extract_image_attributes,
#         description="Use this to extract attributes (objects, colors, style, text) from a local image path."
#     )
# ]

# # A deterministic, zero‐shot agent
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
# agent = initialize_agent(
#     tools,
#     llm,
#     agent="zero-shot-react-description",
#     verbose=True
# )

# if __name__ == "__main__":
#     img_path = "/Users/salmolichandra/Desktop/Agent_pre/Balmain_logo-print_T-shirt_row1_1.jpg"
#     result = agent.invoke(f"ImageAttributeExtractor {img_path}")
#     print(result)






import os
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI

from image_tool import extract_image_attributes

# Folder containing your images; update this path as needed
default_folder = "/Users/salmolichandra/Desktop/Agent_pre/images"

# Define the tool
tools = [
    Tool(
        name="ImageAttributeExtractor",
        func=extract_image_attributes,
        description="Use this to extract attributes (objects, colors, style, text) from a local image path."
    )
]

# Initialize the agent
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

def process_image(path: str):
    """Process a single image file through the agent and print results."""
    print(f"\nProcessing image: {path}")
    result = agent.invoke(f"ImageAttributeExtractor {path}")
    print(result)


def process_folder(folder_path: str):
    """Iterate over all supported image files in a folder."""
    exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(exts):
            file_path = os.path.join(folder_path, filename)
            process_image(file_path)

if __name__ == "__main__":
    # Simply process the default_folder
    process_folder(default_folder)
