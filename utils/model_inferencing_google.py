from google.cloud import aiplatform

# Set your Google Cloud project ID and location
project_id = "your-project-id"
location = "us-central1"  # Change to your desired location

# Initialize aiplatform client
client = aiplatform.gapic.EndpointServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})

# Define the display name for the endpoint
endpoint_display_name = "my-endpoint"

# Define the metadata schema for the endpoint (optional)
metadata_schema = {
    "inputSchema": {"features": {"input": {"type": "float"}}},
    "outputSchema": {"features": {"output": {"type": "float"}}},
}

# Define traffic split for the endpoint (optional)
traffic_split = {"0": 100}

# Create the endpoint
endpoint = client.create_endpoint(
    parent=f"projects/{project_id}/locations/{location}",
    endpoint={"display_name": endpoint_display_name, "metadata_schema": metadata_schema},
    traffic_split=traffic_split,
)

print("Endpoint created:")
print(endpoint)
