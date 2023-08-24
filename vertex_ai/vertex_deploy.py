from google.cloud import aiplatform

# Define your Vertex AI model deployment
def deploy_vertex_model(project_id: str, model_name: str, endpoint_name: str):
    client = aiplatform.gapic.EndpointServiceClient(
        client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    )
    
    model = f"projects/{project_id}/locations/us-central1/models/{model_name}"
    endpoint = f"projects/{project_id}/locations/us-central1/endpoints/{endpoint_name}"
    
    deployment = aiplatform.gapic.DeployedModel(
        model=model,
        display_name=model_name,
    )
    
    traffic_split = {"0": 100}  # Split traffic evenly between models
    
    client.deploy_model(endpoint=endpoint, deployed_model=deployment, traffic_split=traffic_split)

if __name__ == "__main__":
    project_id = "your-project-id"
    model_name = "your-model-name"
    endpoint_name = "your-endpoint-name"

    deploy_vertex_model(project_id, model_name, endpoint_name)
