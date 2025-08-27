# backend/check_methods.py
import sys
from google.cloud import aiplatform_v1beta1
from google.api_core import client_options as client_options_lib

def check_client_methods():
    """Inspects and prints the available methods on the PredictionServiceClient."""
    try:
        location = "us-central1"
        client_options = client_options_lib.ClientOptions(
            api_endpoint=f"{location}-aiplatform.googleapis.com"
        )
        prediction_client = aiplatform_v1beta1.PredictionServiceClient(
            client_options=client_options
        )
        
        print("--- Available methods on PredictionServiceClient (v1beta1) ---")
        # Print public methods, excluding any special/private ones
        methods = [m for m in dir(prediction_client) if not m.startswith('_')]
        for method in sorted(methods):
            print(method)
        print("--- End of methods ---")
        
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    check_client_methods()
