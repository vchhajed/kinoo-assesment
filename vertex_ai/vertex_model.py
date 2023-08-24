from google.cloud import aiplatform

# Define your Vertex AI model
def create_vertex_model(project_id: str, model_display_name: str, container_uri: str):
    client = aiplatform.gapic.ModelServiceClient(client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"})
    model = aiplatform.gapic.Model(
        display_name=model_display_name,
        container_spec={"image_uri": container_uri},
    )

    parent = f"projects/{project_id}/locations/us-central1"
    response = client.create_model(parent=parent, model=model)

    print(f"Created Vertex AI model: {response.name}")

if __name__ == "__main__":
    project_id = "your-project-id"
    model_display_name = "your-model-name"
    container_uri = "your/model/container:latest"

    create_vertex_model(project_id, model_display_name, container_uri)
