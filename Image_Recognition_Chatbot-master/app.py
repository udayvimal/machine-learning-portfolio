import os
import fitz  # PyMuPDF
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP for captioning
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    blip_processor = None
    blip_model = None

# Load classifier (lightweight & open-source)
try:
    classifier_processor = AutoProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    classifier_model = AutoModelForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224").to(device)
except Exception as e:
    print(f"Error loading classifier model: {e}")
    classifier_processor = None
    classifier_model = None

# Expanded list of keywords (you can further customize or expand this list)
fabric_keywords = [
    "cotton", "denim", "leather", "silk", "polyester", "wool", "linen", "velvet", "suede", "nylon", 
    "rayon", "spandex", "cashmere", "jersey", "fleece", "tweed"
]

season_keywords = [
    "summer", "winter", "spring", "fall", "autumn", "monsoon", "rainy", "tropical", "hot", "cold"
]

style_keywords = [
    "casual", "formal", "streetwear", "vintage", "boho", "sporty", "elegant", "chic", "minimalist", 
    "punk", "preppy", "retro", "office wear", "cocktail", "evening wear", "beachwear", "athleisure"
]

# Add fashion accessories and clothing details
accessory_keywords = [
    "sunglasses", "scarf", "hat", "gloves", "belt", "necklace", "bracelet", "watch", "earrings", 
    "bag", "shoes", "boots", "sandals", "flip flops", "purse", "backpack", "tote bag"
]

# Weather and occasion-related keywords
occasion_keywords = [
    "wedding", "party", "casual outing", "work", "gym", "date", "vacation", "travel", "interview"
]

# Function to extract text from a single PDF file
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""

# Function to extract text from all PDFs in a directory
def extract_text_from_books(directory):
    text_data = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            text_data += extract_text_from_pdf(pdf_path)
    return text_data

# Extract text data from all PDFs in the specified directory
book_text_data = extract_text_from_books(r"C:\Users\2k22c\myenv\venv\Image_Recognition_Chatbot\books")

# Function to extract attributes from the caption and text data (learn from PDF content)
def extract_attributes(caption, query, book_data):
    lower_caption = caption.lower()
    response_parts = []

    # Fabric Identification
    if any(word in query.lower() for word in ["fabric", "material"]):
        book_data_lower = book_data.lower()
        for fabric in fabric_keywords:
            if fabric in lower_caption or fabric in book_data_lower:
                response_parts.append(f"The fabric looks like {fabric}.")
                break
        else:
            response_parts.append("Fabric could not be confidently identified.")

    # Season Identification
    if "season" in query.lower():
        for season in season_keywords:
            if season in lower_caption or season in book_data.lower():
                response_parts.append(f"This outfit seems suitable for {season} season.")
                break
        else:
            response_parts.append("No clear seasonal information found.")

    # Style Identification
    if "style" in query.lower():
        for style in style_keywords:
            if style in lower_caption or style in book_data.lower():
                response_parts.append(f"The style appears to be {style}.")
                break
        else:
            response_parts.append("Style could not be determined clearly.")
    
    # Suggest Occasions and Accessories Based on Query
    if any(word in query.lower() for word in occasion_keywords):
        response_parts.append("This outfit would be great for your upcoming occasion!")
    
    if any(word in query.lower() for word in accessory_keywords):
        response_parts.append("Consider adding accessories like a watch, belt, or sunglasses to complete the look!")

    return " ".join(response_parts)

# Function to handle image captioning and attribute extraction
def chat_with_image(image_path, query, history=[]):
    if not all([blip_model, blip_processor, classifier_model, classifier_processor]):
        return "Model or processor not initialized.", history

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return "Error opening image.", history

    try:
        # Step 1: Caption using BLIP
        blip_inputs = blip_processor(image, return_tensors="pt").to(device)
        blip_output = blip_model.generate(**blip_inputs, max_new_tokens=64)
        caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)

        # Step 2: Classification using DeiT
        classifier_inputs = classifier_processor(images=image, return_tensors="pt").to(device)
        classifier_outputs = classifier_model(**classifier_inputs)
        pred_class_idx = classifier_outputs.logits.argmax(-1).item()
        classification = classifier_model.config.id2label[pred_class_idx]

        # Step 3: Extract fashion attributes from the caption and PDF text
        attributes = extract_attributes(caption, query, book_text_data)

        # Compose response
        full_response = (
            f"🖼️ Image Caption: {caption}\n"
            f"🔍 Classification: {classification}\n"
            f"🤖 Query: {query}\n"
            f"💬 Response: {attributes if attributes else 'This appears to be a stylish ' + classification.lower()}."
        )

        # Update chat history
        history.append({"role": "user", "content": {"type": "text", "text": query}})
        history.append({"role": "assistant", "content": {"type": "text", "text": full_response}})

        return full_response, history

    except Exception as e:
        print(f"Error during processing: {e}")
        return "An error occurred during processing.", history

# Main driver to test the system
if __name__ == "__main__":
    image_path = "C:/Users/2k22c/myenv/venv/Image_Recognition_Chatbot/Image_Recognition_Chatbot/swapnil2a.jpg"
    conversation_history = [] 
    query = "What is the fabric and style of this outfit?"
    response, conversation_history = chat_with_image(image_path, query, conversation_history)
    print(response)