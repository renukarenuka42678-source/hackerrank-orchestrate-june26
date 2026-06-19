import os
import json
import pandas as pd
from google import genai  # Example using Gemini client, adapt to your preferred model API

# Initialize your AI Client
# Make sure your API key is set in your environment variables (e.g., export GEMINI_API_KEY="your-key")
client = genai.Client()

def analyze_evidence_with_ai(conversation, object_type, user_history, image_paths):
    """
    Sends the claim details and associated images to a Multi-Modal VLM.
    Returns a structured dictionary matching the output schema.
    """
    
    # 1. Prepare images for the API
    uploaded_images = []
    for path in image_paths:
        if os.path.exists(path):
            # Load the image using your model's preferred utility
            img = genai.types.Part.from_bytes(
                data=open(path, "rb").read(),
                mime_type="image/jpeg"
            )
            uploaded_images.append(img)

    # 2. Craft a strict system prompt instructing the model to return JSON
    prompt = f"""
    You are an expert multi-modal claims adjuster AI.
    
    [CLAIM CONTEXT]
    - Object Type: {object_type}
    - Conversation: {conversation}
    - User Risk History: {user_history}
    
    [TASK]
    Analyze the text context against the attached images. 
    You must output a valid JSON object matching this schema exactly:
    {{
        "decision": "SUPPORTED" or "CONTRADICTED" or "INSUFFICIENT_INFO",
        "visible_issue_type": "string describing damage",
        "relevant_object_part": "string describing part",
        "supporting_image_ids": "comma separated string of filenames used as evidence",
        "severity": "LOW" or "MEDIUM" or "HIGH",
        "risk_flags": "quality" or "mismatch" or "authenticity" or "history" or "none",
        "justification": "A short, fact-grounded sentence explaining your decision."
    }}
    
    Ensure your decision matches these rules:
    - SUPPORTED: Visual evidence matches the claim conversation exactly.
    - CONTRADICTED: Visual evidence directly conflicts with the conversation (e.g., no damage visible, completely different item).
    - INSUFFICIENT_INFO: Images are missing, too blurry, or don't show the claimed area.
    """

    try:
        # 3. Call the model (using a multi-modal capable model like gemini-1.5-flash)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt, *uploaded_images],
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Parse the JSON response
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Error calling model: {e}")
        # Return fallback safe values if the API fails for a row
        return {
            "decision": "INSUFFICIENT_INFO",
            "visible_issue_type": "unknown",
            "relevant_object_part": "unknown",
            "supporting_image_ids": "",
            "severity": "LOW",
            "risk_flags": "none",
            "justification": "Failed to process image evidence due to technical error."
        }

def process_all_claims():
    # Load test data
    test_df = pd.read_csv("dataset/test.csv")
    output_rows = []
    
    print(f"Starting pipeline for {len(test_df)} claims...")

    for index, row in test_df.iterrows():
        claim_id = row['claim_id']
        conversation = row['conversation']
        object_type = row['object_type']
        user_history = row.get('user_history', 'No prior history')
        
        # Locate local images for this specific claim
        # Assuming folder structure inside extracted claims.zip is dataset/images/{claim_id}/
        image_dir = f"dataset/images/{claim_id}"
        image_paths = []
        if os.path.exists(image_dir):
            image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"[{index+1}/{len(test_df)}] Processing Claim ID: {claim_id} with {len(image_paths)} images...")
        
        # Get AI Assessment
        ai_result = analyze_evidence_with_ai(conversation, object_type, user_history, image_paths)
        
        # Gather filenames safely for the supporting_image_ids column
        img_filenames = ", ".join([os.path.basename(p) for p in image_paths]) if image_paths else ""
        
        # Build output format safely matching problem_statement.md
        output_rows.append({
            "claim_id": claim_id,
            "decision": ai_result.get("decision", "INSUFFICIENT_INFO"),
            "visible_issue_type": ai_result.get("visible_issue_type", "none"),
            "relevant_object_part": ai_result.get("relevant_object_part", "none"),
            "supporting_image_ids": ai_result.get("supporting_image_ids", img_filenames),
            "severity": ai_result.get("severity", "LOW"),
            "risk_flags": ai_result.get("risk_flags", "none"),
            "justification": ai_result.get("justification", "No justification generated.")
        })
        
    # Save target CSV
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv("output.csv", index=False)
    print("✨ Execution finished! output.csv generated successfully.")

if __name__ == "__main__":
    process_all_claims()
