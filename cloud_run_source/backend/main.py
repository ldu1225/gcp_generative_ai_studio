import os
import uuid
import json
import logging
import base64
import time
import requests
import google.auth
import google.auth.transport.requests
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from google.cloud import storage, aiplatform, vision, texttospeech

# SDK Imports
import vertexai
from vertexai.preview.generative_models import GenerativeModel as VertexGenerativeModel, Part as VertexPart, GenerationConfig
from vertexai.preview.vision_models import Image, ImageGenerationModel
from vertexai.preview.vision_models import RawReferenceImage, MaskReferenceImage, SubjectReferenceImage, ControlReferenceImage

# Explicitly import the Google AI Generative AI SDK and its types
import google.generativeai as genai
from google.generativeai import types

from PIL import Image as PILImage, ImageDraw, ImageFont
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import traceback
from vertexai.preview.generative_models import HarmCategory, HarmBlockThreshold

# --- 기본 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')
PROJECT_ID = "duleetest"
BUCKET_NAME = "duleetest-hsad-demo-assets"
LOCATION = "us-central1"

# --- 모델 ID ---
GEMINI_PLANNER_MODEL = "gemini-2.5-flash"
IMAGEN_EDIT_MODEL_ID = "imagen-3.0-capability-001"
IMAGEN_GENERATE_MODEL_ID = "imagen-4.0-generate-001"
IMAGEN_UPSCALING_MODEL_ID = "imagen-3.0-generate-002"
VEO_MODEL_ID = "veo-3.0-generate-001"
LYRIA_MODEL_ID = "publishers/google/models/lyria-002"
NANO_BANANA_MODEL_ID = "gemini-2.5-flash-image-preview" # AI Studio Model

# --- 전역 클라이언트 초기화 ---
storage_client = None
gemini_planner = None
edit_model = None
upscale_model = None
generate_model = None
video_model = None
vision_client = None
tts_client = None
prediction_client = None
# This model is for the non-Vertex, Google AI Studio model
nano_banana_model = None 

def init_clients():
    global storage_client, gemini_planner, edit_model, upscale_model, generate_model, video_model, vision_client, tts_client, prediction_client, nano_banana_model
    try:
        # --- Vertex AI Models Initialization ---
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        storage_client = storage.Client(project=PROJECT_ID)
        gemini_planner = VertexGenerativeModel(GEMINI_PLANNER_MODEL)
        edit_model = ImageGenerationModel.from_pretrained(IMAGEN_EDIT_MODEL_ID)
        upscale_model = ImageGenerationModel.from_pretrained(IMAGEN_UPSCALING_MODEL_ID)
        generate_model = ImageGenerationModel.from_pretrained(IMAGEN_GENERATE_MODEL_ID)
        video_model = VertexGenerativeModel(VEO_MODEL_ID)
        vision_client = vision.ImageAnnotatorClient()
        tts_client = texttospeech.TextToSpeechClient()
        prediction_client = aiplatform.gapic.PredictionServiceClient(client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})
        log.info("All Vertex AI models and GCS clients initialized successfully.")

        # --- AI Studio (Non-Vertex) Model Initialization using API Key ---
        try:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set for Nano-banana model.")
            # Configure the genai SDK with the API key
            genai.configure(api_key=api_key)
            # Initialize the model using the high-level GenerativeModel class
            nano_banana_model = genai.GenerativeModel(NANO_BANANA_MODEL_ID)
            log.info("Nano-banana (AI Studio) model initialized successfully using API Key.")
        except Exception as e:
            nano_banana_model = None
            # Log the full traceback for better debugging
            log.error(f"Failed to initialize Nano-banana model. API will be unavailable. Reason: {e}", exc_info=True)
            # Re-raise the exception to prevent the application from starting with a broken model
            # This will cause the Cloud Run instance to fail startup, which is desirable
            # if a critical component like a model cannot be initialized.
            raise

    except Exception as e:
        log.critical(f"A critical error occurred during Google Cloud services initialization: {e}", exc_info=True)
        raise

def upload_to_gcs(client, bucket_name, file_stream, content_type, prefix):
    if not client:
        raise ConnectionError("GCS client is not initialized.")
    try:
        bucket = client.bucket(bucket_name)
        ext = content_type.split('/')[-1] if '/' in content_type else 'png'
        filename = f"{prefix}/{uuid.uuid4().hex}.{ext}"
        blob = bucket.blob(filename)
        file_stream.seek(0)
        blob.upload_from_file(file_stream, content_type=content_type)
        blob.make_public()
        log.info(f"File uploaded to GCS. Public URL: {blob.public_url}")
        return blob.public_url, f"gs://{bucket_name}/{filename}"
    except Exception as e:
        log.error(f"GCS upload failed: {e}", exc_info=True)
        raise

# ... (rest of the Vertex AI functions remain unchanged)
def call_gemini_for_editing_plan(gcs_uri: str, user_prompt: str) -> dict:
    log.info(f"Gemini Planner ({GEMINI_PLANNER_MODEL}): Generating detailed editing plan.")
    org_image = VertexPart.from_uri(gcs_uri, mime_type="image/png")
    prompt_template = f"""
        You're an advertising professional utilizing Imagen for ad creation. 
        Generate an English Imagen prompt that will transform the provided image to meet the user's specifications.
        User Request: {user_prompt} 
        <instructions>
        <task1> Analyze the original photo and write a detailed description of it including photo or painting style (within 60 tokens). </task1>
        <task2> Identify and describe the most central object in the original photo (within 20 tokens). </task2>
        <task3> 
        Analyze the user request and identify the edit type:
        * SUBJECT_EDITING: Keep the main object itself but change its position,its motion, background, color of background, or anything besides of main object. 
        * CONTROLLED_EDITING: Maintain the overall image composition but change color, tone, or convert canny map to image. 
        * INSTRUCT_EDITING: Change overall image composition and pating style or cartoon style. 
        * EDIT_MODE_DEFAULT: Default mode for general edits.
        Store this information in the `edit_type` variable. 
        </task3>
        <task4> Clearly explain the relationship between the user request and the extracted main object or background. Add this explanation to the edit_mode_selection_reason field. </task4>
        <task5> Determine the edit mode: inpainting-insert, inpainting-remove, outpainting, default. Store this in the edit_mode variable. </task5>
        <task6> Determine the mask mode: NONE, MASK_MODE_FOREGROUND, MASK_MODE_BACKGROUND, MASK_MODE_DEFAULT. Store this in the mask_mode variable. </task6>
        <task7> Determine the subject type for SUBJECT_EDITING: person, animal, product, default. Store this in the subject_type variable. </task7>
        <task8> Write a imagen positive_prompt in English. </task8>
        <task9> Determine guidance_scale (e.g., 1.0 for foreground, 20.0 for background). </task9>
        <task10> Determine mask_dilation (e.g., 0.005 for minimal, 0.03 for significant). </task10>
        <task11> Determine control_type: CONTROL_TYPE_SCRIBBLE, CONTROL_TYPE_CANNY. </task11>
        </instructions>
        <output>
        {{
        "org_image_description" : "...", "main_object_description" : "...", "edit_type" : "...", "edit_mode_selection_reason": "...", "edit_mode" : "...", "mask_mode" : "...", "subject_type" : "...", "positive_prompt" : "...", "negative_prompt" : "...", "guidance_scale" : "...", "mask_dilation" : "...", "control_type" : "..."
        }}
        </output>
        """
    response_schema = {"type" : "OBJECT", "properties" : {"org_image_description": {"type": "STRING"}, "main_object_description": {"type": "STRING"}, "edit_type": {"type": "STRING"}, "edit_mode_selection_reason": {"type": "STRING"}, "edit_mode": {"type": "STRING"}, "mask_mode": {"type": "STRING"}, "subject_type": {"type": "STRING"}, "positive_prompt": {"type": "STRING"}, "negative_prompt": {"type": "STRING"}, "guidance_scale": {"type": "STRING"}, "mask_dilation": {"type": "STRING"}, "control_type": {"type": "STRING"}}}
    response = gemini_planner.generate_content(
        [org_image, prompt_template],
        generation_config=GenerationConfig(response_mime_type="application/json", response_schema=response_schema)
    )    
    json_response = json.loads(response.text)
    log.info(f"Generated detailed plan: {json_response}")
    return json_response

def execute_imagen_edit(gcs_uri: str, plan: dict):
    log.info(f"--- Starting Imagen Edit (Model: {IMAGEN_EDIT_MODEL_ID}, Type: {plan.get('edit_type')}) ---")
    org_image = Image(gcs_uri=gcs_uri)
    raw_ref_image = RawReferenceImage(image=org_image, reference_id=0)
    
    edit_type = plan.get("edit_type")
    positive_prompt = plan.get("positive_prompt")
    negative_prompt = plan.get("negative_prompt")
    guidance_scale = float(plan.get("guidance_scale", 1.0))
    
    if edit_type == "EDIT_MODE_DEFAULT":
        edit_mode = plan.get("edit_mode")
        if edit_mode == "default":
            return edit_model.edit_image(prompt=positive_prompt, negative_prompt=negative_prompt, edit_mode=edit_mode, reference_images=[raw_ref_image], number_of_images=1, guidance_scale=guidance_scale)
        else:
            mask_mode = plan.get("mask_mode")
            mask_dilation = float(plan.get("mask_dilation", 0.0))
            if mask_mode == "MASK_MODE_FOREGROUND":
                mask_ref_image = MaskReferenceImage(reference_id=1, image=org_image, mask_mode="foreground", dilation=mask_dilation) 
            elif mask_mode == "MASK_MODE_BACKGROUND":
                mask_ref_image = MaskReferenceImage(reference_id=1, image=org_image, mask_mode="background", dilation=mask_dilation)
            else:
                mask_ref_image = MaskReferenceImage(reference_id=1, image=org_image, mask_mode="default", dilation=mask_dilation)
            return edit_model.edit_image(prompt=positive_prompt, negative_prompt=negative_prompt, edit_mode=edit_mode, reference_images=[raw_ref_image, mask_ref_image], number_of_images=1, guidance_scale=guidance_scale)
    elif edit_type == "SUBJECT_EDITING":
        subject_reference_image = SubjectReferenceImage(reference_id=1, image=org_image, subject_description=plan.get("main_object_description"), subject_type=plan.get("subject_type"))
        return edit_model._generate_images(prompt=positive_prompt, negative_prompt=negative_prompt, number_of_images=1, reference_images=[subject_reference_image], guidance_scale=guidance_scale)
    elif edit_type == "CONTROLLED_EDITING":
        control_image = ControlReferenceImage(reference_id=1, image=org_image, control_type=plan.get("control_type", "canny"), enable_control_image_computation=True)
        return edit_model._generate_images(prompt=f"Generate an image using the control reference image [1] to match the description: {positive_prompt}", negative_prompt=negative_prompt, number_of_images=1, reference_images=[control_image], guidance_scale=guidance_scale)
    elif edit_type == "INSTRUCT_EDITING":
        return edit_model._generate_images(prompt=positive_prompt, negative_prompt=negative_prompt, number_of_images=1, reference_images=[raw_ref_image])
    else:
        raise ValueError(f"Unsupported edit_type: {edit_type}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/api/edit", methods=["POST"])
def edit_image_api():
    if 'image' not in request.files or 'prompt' not in request.form: return jsonify({"error": "Image or prompt missing"}), 400
    try:
        image_bytes_io = BytesIO(request.files['image'].read())
        original_url, gcs_uri = upload_to_gcs(storage_client, BUCKET_NAME, image_bytes_io, 'image/png', 'originals')
        plan = call_gemini_for_editing_plan(gcs_uri, request.form['prompt'])
        edited_images = execute_imagen_edit(gcs_uri, plan)
        if not edited_images: return jsonify({"error": "Image could not be edited."} ), 400
        buffer = BytesIO()
        edited_images[0]._pil_image.save(buffer, format="PNG")
        edited_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, buffer, 'image/png', 'edited')
        return jsonify({"originalImageUrl": original_url, "editedImageUrl": edited_url})
    except Exception as e:
        log.error(f"CRITICAL ERROR in edit_image_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route("/api/edit_nanobanana", methods=["POST"])
def edit_image_nanobanana_api():
    if not nano_banana_model:
        log.error("Nano-banana model is not available, aborting request.")
        return jsonify({"error": "The Nano-banana editing service is currently unavailable. Check server logs for API Key or initialization issues."}, 503)

    if 'images' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Images or prompt missing"}), 400
    
    try:
        image_files = request.files.getlist('images')
        user_prompt = request.form['prompt']

        if not image_files:
            return jsonify({"error": "At least one image is required"}), 400

        first_image_bytes = image_files[0].read()
        image_files[0].seek(0)
        original_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, BytesIO(first_image_bytes), image_files[0].mimetype, 'originals_nano')

        log.info(f"Calling AI Studio model ({NANO_BANANA_MODEL_ID}) with {len(image_files)} images using genai.GenerativeModel.")
        
        instructional_prompt = "Generate an image based on the user's text prompt and the provided images. Your ONLY output should be the resulting image. User's request: " + user_prompt
        
        model_contents = [instructional_prompt]
        for image_file in image_files:
            img = PILImage.open(image_file.stream)
            model_contents.append(img)

        safety_settings = {
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        }
        
        response_chunks = nano_banana_model.generate_content(
            model_contents,
            stream=True,
            safety_settings=safety_settings
        )
        
        edited_image_bytes = None
        text_parts = []
        
        try:
            for chunk in response_chunks:
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    error_message = f"Request was blocked for safety reasons: {chunk.prompt_feedback.block_reason_message or chunk.prompt_feedback.block_reason}"
                    log.error(error_message)
                    return jsonify({"error": error_message}), 400

                if not chunk.parts:
                    continue

                for part in chunk.parts:
                    if part.inline_data and "image" in part.inline_data.mime_type:
                        edited_image_bytes = part.inline_data.data
                        break
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                
                if edited_image_bytes:
                    break
        except Exception as e:
            log.warning(f"An error occurred during model response streaming: {e}", exc_info=True)
            # This allows the code to proceed to the check for partial data
            pass

        if edited_image_bytes:
            buffer = BytesIO(edited_image_bytes)
            edited_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, buffer, 'image/png', 'edited_nano')
            return jsonify({"originalImageUrl": original_url, "editedImageUrl": edited_url})
        
        elif text_parts:
            full_text_response = "".join(text_parts).strip()
            log.warning(f"Model returned a text response instead of an image: '{full_text_response}'")
            return jsonify({"error": f"Model returned a text response instead of an image: '{full_text_response}'"}), 400
        
        else:
            log.error("Model response did not contain any valid image or text data after streaming.")
            return jsonify({"error": "Image could not be edited. The model returned an empty response."}, 500)

    except Exception as e:
        log.error(f"CRITICAL ERROR in edit_image_nanobanana_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route("/api/enhance_prompt", methods=["POST"])
def enhance_prompt_api():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is missing"}), 400
    try:
        original_prompt = data['prompt']
        enhancement_instruction = f"Make the following prompt more descriptive and artistic for an AI image generator, in English: '{original_prompt}'"
        
        response = gemini_planner.generate_content(enhancement_instruction)
        
        enhanced_prompt = response.text.strip()
        
        log.info(f"Enhanced prompt from '{original_prompt}' to '{enhanced_prompt}'")
        return jsonify({"enhancedPrompt": enhanced_prompt})
    except Exception as e:
        log.error(f"CRITICAL ERROR in enhance_prompt_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


@app.route("/api/upscale_image", methods=["POST"])
def upscale_image_api():
    log.info(f"--- Received new request on /api/upscale_image (Stable Version with Debugging) ---")
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required"}), 400
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        original_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, BytesIO(image_bytes), image_file.mimetype, 'originals_for_upscale')

        log.info(f"Step 1: Initializing source image for model {IMAGEN_UPSCALING_MODEL_ID}.")
        source_image = Image(image_bytes=image_bytes)

        log.info("Step 2: Calling upscale_image API to get the upscaled image data back directly.")
        upscaled_image = upscale_model.upscale_image(
            image=source_image,
            upscale_factor="x4"
        )
        
        if not upscaled_image or not hasattr(upscaled_image, '_pil_image'):
             log.error("Upscaling process did not return a valid image object.")
             raise ValueError("Upscaling process failed to return a valid image.")

        log.info("Step 3: Upscaling successful. Inspecting returned image before upload.")
        log.info(f"DEBUG: Returned image object type: {type(upscaled_image)}")
        log.info(f"DEBUG: PIL image details: size={upscaled_image._pil_image.size}, mode={upscaled_image._pil_image.mode}")

        buffer = BytesIO()
        upscaled_image._pil_image.save(buffer, format="PNG")
        
        buffer.seek(0, os.SEEK_END)
        buffer_size = buffer.tell()
        buffer.seek(0)
        log.info(f"DEBUG: Buffer size after saving PIL image: {buffer_size} bytes")

        if buffer_size == 0:
            raise ValueError("Image buffer is empty after saving, indicating a problem with the upscaled image data.")

        log.info("Step 4: Uploading buffer to GCS.")
        upscaled_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, buffer, 'image/png', 'upscaled')
        
        log.info(f"Successfully uploaded upscaled image to public URL: {upscaled_url}")
        
        return jsonify({"originalUrl": original_url, "upscaledUrl": upscaled_url})
    except Exception as e:
        log.error(f"CRITICAL ERROR in upscale_image_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route("/api/generate_image", methods=["POST"])
def generate_image_api():
    data = request.get_json()
    if not data or 'prompt' not in data: return jsonify({"error": "Prompt missing"}), 400
    try:
        images = generate_model.generate_images(prompt=data['prompt'], number_of_images=1)
        if not images: return jsonify({"error": "Image generation failed."} ), 400
        buffer = BytesIO()
        images[0]._pil_image.save(buffer, format="PNG")
        generated_url, gcs_uri = upload_to_gcs(storage_client, BUCKET_NAME, buffer, 'image/png', 'generated')
        buffer.seek(0)
        image_data_url = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
        return jsonify({"imageUrl": generated_url, "gcsUri": gcs_uri, "imageData": image_data_url})
    except Exception as e:
        log.error(f"CRITICAL ERROR in generate_image_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route("/api/simulate_finetuning", methods=["POST"])
def simulate_finetuning_api():
    log.info("--- Received new request on /api/simulate_finetuning (prompt-style merging logic) ---")
    if 'prompt' not in request.form or 'files' not in request.files:
        return jsonify({"error": "Prompt or files missing"}), 400
    try:
        files = request.files.getlist('files')
        user_prompt = request.form['prompt']
        style_image_parts = []
        for file in files:
            _, gcs_uri = upload_to_gcs(storage_client, BUCKET_NAME, file.stream, file.mimetype, 'style_references')
            style_image_parts.append(VertexPart.from_uri(gcs_uri, mime_type=file.mimetype))
        
        style_merging_prompt = [
            "You are a creative prompt engineer for an AI image generator.",
            "Combine the user's subject with the artistic style from the reference images into a single, detailed, and effective prompt.",
            "Analyze the style (color, texture, mood, composition) of the reference images and blend it seamlessly with the user's subject.",
            "Do not describe the content of the reference images, only their style.",
            "The final output must be a single, cohesive prompt in English that the image generator can directly use.",
            f"User's subject prompt: '{user_prompt}'",
        ] + style_image_parts

        response = gemini_planner.generate_content(style_merging_prompt)
        final_prompt = response.text.strip().replace("\n", " ")
        log.info(f"Generated a new combined prompt: {final_prompt}")

        images = generate_model.generate_images(prompt=final_prompt, number_of_images=1)
        if not images: raise ValueError("Image generation failed with the combined prompt.")
        
        buffer = BytesIO()
        images[0]._pil_image.save(buffer, format="PNG")
        image_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, buffer, 'image/png', 'finetuned')

        return jsonify({"imageUrl": image_url, "styleDna": final_prompt})
    except Exception as e:
        log.error(f"CRITICAL ERROR in simulate_finetuning_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


@app.route("/api/generate_video", methods=["POST"])
def generate_video_api():
    log.info("--- Received new request on /api/generate_video (Corrected REST Polling) ---")
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image or prompt missing"}), 400
    
    try:
        image_file = request.files['image']
        prompt_text = request.form['prompt']

        # --- FIX START: Validate and process the image ---
        try:
            with PILImage.open(image_file.stream) as img:
                # Ensure image is in a web-friendly format like PNG or JPEG
                output_format = "PNG" if img.format != "JPEG" else "JPEG"
                mime_type = f"image/{output_format.lower()}"
                
                # Re-save the image to a buffer to ensure it's clean and valid
                buffer = BytesIO()
                img.save(buffer, format=output_format)
                image_bytes = buffer.getvalue()

                if not image_bytes:
                    raise ValueError("Image data is empty after processing.")

        except (IOError, SyntaxError) as e:
            log.error(f"Invalid image file uploaded: {e}", exc_info=True)
            return jsonify({"error": "Invalid or corrupt image file provided. Please upload a valid image."}, 400)
        # --- FIX END ---

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # 1. Programmatically get the authentication token
        credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        auth_token = credentials.token
        
        # 2. Construct the API endpoint and headers
        api_endpoint = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{VEO_MODEL_ID}:predictLongRunning"
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }

        # 3. Construct the request payload
        payload = {
            "instances": [
                {
                    "prompt": prompt_text,
                    "image": {
                        "bytesBase64Encoded": encoded_image,
                        "mimeType": mime_type
                    }
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "durationSeconds": 8
            }
        }

        # 4. Make the initial POST request to start the LRO
        log.info(f"Making POST request to start LRO at {api_endpoint}")
        response = requests.post(api_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        operation = response.json()
        operation_name = operation.get("name")
        if not operation_name:
            raise ValueError(f"Failed to start LRO. Response: {operation}")
            
        log.info(f"Video generation LRO started. Operation name: {operation_name}")

        # 5. Poll the operation status using the correct fetchPredictOperation endpoint
        fetch_endpoint = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{VEO_MODEL_ID}:fetchPredictOperation"
        
        while True:
            time.sleep(10)
            log.info(f"Polling operation status for {operation_name}...")
            poll_payload = {"operationName": operation_name}
            op_response = requests.post(fetch_endpoint, headers=headers, json=poll_payload)
            op_response.raise_for_status()
            op_status = op_response.json()

            if op_status.get("done"):
                log.info("LRO completed. Full response object:")
                # log.info(json.dumps(op_status, indent=2))

                if "error" in op_status:
                    error_details = op_status["error"]
                    error_message = error_details.get('message', '')
                    log.error(f"LRO finished with error: {error_details}")
                    if "Responsible AI practices" in error_message:
                        return jsonify({"error": error_message}), 400
                    raise Exception(f"Video generation failed: {error_message}")
                
                # Correctly parse the nested response
                video_response = op_status.get("response", {})
                video_list = video_response.get("videos", [])

                if not video_list:
                    log.error(f"Veo model did not return any videos. Full API response: {json.dumps(op_status, indent=2)}")
                    return jsonify({"error": "Video generation failed: The AI model could not generate a video. This might be due to the input image or prompt violating usage guidelines, or an internal model issue. Please try with a different image or prompt."}, 400)
                
                # Extract video data from the correct path
                video_b64 = video_list[0].get("bytesBase64Encoded")
                if not video_b64:
                    log.error(f"Veo model returned an empty video data. Full video list: {json.dumps(video_list, indent=2)}")
                    return jsonify({"error": "Video generation failed: The AI model returned an empty video. This might be due to an internal model issue. Please try again."}, 500)

                video_bytes = base64.b64decode(video_b64)
                video_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, BytesIO(video_bytes), 'video/mp4', 'generated_videos_rest')
                
                log.info(f"Successfully generated and uploaded video: {video_url}")
                return jsonify({"videoUrl": video_url})
                
    except requests.exceptions.HTTPError as e:
        error_message = f"HTTP ERROR in generate_video_api: {e.response.text}"
        log.error(error_message, exc_info=True)
        return jsonify({"error": f"An API error occurred: {e.response.text}"}), e.response.status_code
    except Exception as e:
        error_message = f"CRITICAL ERROR in generate_video_api: {e}"
        log.error(error_message, exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500



@app.route("/api/generate_music", methods=["POST"])
def generate_music_api():
    data = request.get_json()
    if not data or 'prompt' not in data: return jsonify({"error": "Prompt missing"}), 400
    try:
        user_prompt = data['prompt']
        translation_prompt = f"Translate the following user request for music into a simple, descriptive English prompt for an AI music generation model. Focus on instruments, mood, and tempo. User request: '{user_prompt}'"
        translated_response = gemini_planner.generate_content(translation_prompt)
        english_prompt = translated_response.text.strip()
        
        instance = json_format.ParseDict({"prompt": english_prompt}, Value())
        response = prediction_client.predict(endpoint=f"projects/{PROJECT_ID}/locations/{LOCATION}/{LYRIA_MODEL_ID}", instances=[instance])
        b64_audio = dict(response.predictions[0])['bytesBase64Encoded']
        audio_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, BytesIO(base64.b64decode(b64_audio)), 'audio/wav', 'music')
        return jsonify({"audioUrl": audio_url})
    except Exception as e:
        log.error(f"CRITICAL ERROR in generate_music_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route("/api/generate_voice", methods=["POST"])
def generate_voice_api():
    data = request.get_json()
    if not data or 'prompt' not in data or 'voice' not in data: return jsonify({"error": "Prompt or voice missing"}), 400
    try:
        synthesis_input = texttospeech.SynthesisInput(text=data['prompt'])
        voice_params = texttospeech.VoiceSelectionParams(language_code=data['voice'].split('-')[0] + '-' + data['voice'].split('-')[1], name=data['voice'])
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        audio_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, BytesIO(response.audio_content), 'audio/mp3', 'voices')
        return jsonify({"audioUrl": audio_url})
    except Exception as e:
        log.error(f"CRITICAL ERROR in generate_voice_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route("/api/analyze_image", methods=["POST"])
def analyze_image_api():
    data = request.get_json()
    if not data or 'gcsUri' not in data: return jsonify({"error": "GCS URI missing"}), 400
    
    gcs_uri = data['gcsUri']
    
    try:
        image = vision.Image(source=vision.ImageSource(image_uri=gcs_uri))
        features = [
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=10),
            vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
            vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
        ]
        response = vision_client.annotate_image(request={'image': image, 'features': features})

        objects = [{'name': obj.name, 'vertices': [(v.x, v.y) for v in obj.bounding_poly.normalized_vertices]} for obj in response.localized_object_annotations]
        safety = {
            'adult': vision.Likelihood(response.safe_search_annotation.adult).name,
            'spoof': vision.Likelihood(response.safe_search_annotation.spoof).name,
            'medical': vision.Likelihood(response.safe_search_annotation.medical).name,
            'violence': vision.Likelihood(response.safe_search_annotation.violence).name,
            'racy': vision.Likelihood(response.safe_search_annotation.racy).name,
        }
        text = response.text_annotations[0].description if response.text_annotations else ""

        log.info("Performing advanced analysis with Gemini.")
        image_part = VertexPart.from_uri(gcs_uri, mime_type="image/png")
        
        gemini_prompt = """
        Analyze the provided image for advertising suitability. Respond in JSON format.
        1.  **labels**: Provide 5-7 specific, single-word descriptive labels. For each label, provide a Korean and English version. Example format: `{\"ko\": \"노인\", \"en\": \"Senior\"}`.
        2.  **analysis_ko**: Provide a brief safety and content assessment in Korean. Analyze potential issues like violence, adult content, or controversial topics. Start with a clear recommendation: '광고에 적합함', '주의 필요', or '부적합'.
        """
        
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "labels": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "ko": {"type": "STRING"},
                            "en": {"type": "STRING"}
                        }
                    }
                },
                "analysis_ko": {"type": "STRING"}
            }
        }

        gemini_response = gemini_planner.generate_content(
            [gemini_prompt, image_part],
            generation_config=GenerationConfig(response_mime_type="application/json", response_schema=response_schema)
        )
        
        gemini_result = json.loads(gemini_response.text)
        gemini_labels = gemini_result.get("labels", [])
        gemini_analysis_text = gemini_result.get("analysis_ko", "")
        log.info(f"Gemini analysis result: {gemini_result}")

        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        image_bytes = blob.download_as_bytes()
        img = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=36)
        except IOError:
            log.warning("DejaVuSans.ttf not found. Using default font.")
            font = ImageFont.load_default()

        COLORS = ['#FF3838', '#FF9D42', '#2ECC71', '#3498DB', '#9B59B6', '#F1C40F']
        for i, obj in enumerate(objects):
            color = COLORS[i % len(COLORS)]
            box = [obj['vertices'][0][0] * width, obj['vertices'][0][1] * height, obj['vertices'][2][0] * width, obj['vertices'][2][1] * height]
            draw.rectangle(box, outline=color, width=5)
            text_position = (box[0] + 5, box[1] + 5)
            text_content = obj['name']
            text_bbox = draw.textbbox(text_position, text_content, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text(text_position, text_content, fill="white", font=font)

        buffer = BytesIO()
        img.save(buffer, "PNG")
        annotated_url, _ = upload_to_gcs(storage_client, BUCKET_NAME, buffer, 'image/png', 'annotated')

        hsad_policy_definition = {
            "주류/담배": "주류 또는 담배와 관련된 콘텐츠는 금지됩니다.",
            "과도한 신체 노출": "선정적인 신체 노출은 허용되지 않습니다.",
            "폭력성": "폭력적이거나 잔인한 묘사는 금지됩니다."
        }
        status = "승인"
        reason = "콘텐츠가 HSAD 광고 정책을 준수합니다."

        if safety['adult'] in ['LIKELY', 'VERY_LIKELY'] or safety['racy'] in ['LIKELY', 'VERY_LIKELY']:
            status = "반려"; reason = "과도한 신체 노출 관련 정책 위반 가능성이 있습니다."
        if safety['violence'] in ['LIKELY', 'VERY_LIKELY']:
            status = "반려"; reason = "폭력성 관련 콘텐츠 정책 위반 가능성이 있습니다."

        prohibited_keywords = ["담배", "흡연", "주류", "술", "맥주", "소주", "폭력", "cigarette", "smoking", "alcohol", "beer", "violence"]
        for label_pair in gemini_labels:
            if any(keyword in label_pair['ko'].lower() or keyword in label_pair['en'].lower() for keyword in prohibited_keywords):
                status = "반려"
                reason = f"Gemini 분석 결과, 정책에 위배되는 '{label_pair['ko']}({label_pair['en']})' 키워드가 감지되었습니다."
                break 

        return jsonify({
            "objects": objects, "safety": safety, "text": text,
            "annotatedImageUrl": annotated_url,
            "hsadPolicyDefinition": hsad_policy_definition,
            "hsadPolicy": {"status": status, "reason": reason},
            "geminiLabels": gemini_labels,
            "geminiAnalysis": gemini_analysis_text
        })

    except Exception as e:
        log.error(f"CRITICAL ERROR in analyze_image_api: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    init_clients()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
