import base64
import logging
from io import BytesIO
from PIL import Image as PILImage
from google.cloud import aiplatform_v1beta1
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.api_core import client_options as client_options_lib

# --- Configuration ---
PROJECT_ID = "duleetest"
LOCATION = "us-central1"
VEO_MODEL_ID = "veo-3.0-generate-001"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def final_predict_test():
    """
    Final test for video generation using the PredictionServiceClient's `predict` method.
    """
    try:
        # 1. Initialize the v1beta1 client
        client_options = client_options_lib.ClientOptions(api_endpoint=f"{LOCATION}-aiplatform.googleapis.com")
        prediction_client = aiplatform_v1beta1.PredictionServiceClient(client_options=client_options)

        # 2. Create a test image
        log.info("Creating a test image...")
        img = PILImage.new('RGB', (256, 256), color='magenta')
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        log.info("Test image created and encoded.")

        # 3. Prepare the request payload
        prompt_text = "A magenta square."
        instance_dict = {
            "prompt": prompt_text,
            "image": {"bytesBase64Encoded": encoded_image},
        }
        instance = json_format.ParseDict(instance_dict, Value())
        
        endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{VEO_MODEL_ID}"

        # 4. Call the `predict` method
        log.info(f"Calling the `predict` method on endpoint: {endpoint}")
        response = prediction_client.predict(endpoint=endpoint, instances=[instance])
        
        # 5. Process the result
        if not response.predictions:
            raise ValueError("Prediction response is empty.")
            
        video_b64 = response.predictions[0]
        video_bytes = base64.b64decode(video_b64)

        if not video_bytes:
            raise ValueError("Decoded video data is empty.")

        # 6. Save the output
        output_filename = "final_predict_test_output.mp4"
        with open(output_filename, "wb") as f:
            f.write(video_bytes)
        log.info(f"SUCCESS: Video saved to {output_filename}")

    except Exception as e:
        log.error(f"FAILURE: An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    final_predict_test()
