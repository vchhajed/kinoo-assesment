# kinoo-assesment

## source code (src folder)
- data: Here, the structure of the data is mentioned. We would require storage to store our data (tools: S3, GCS)
- models: Here, the structure of the model is mentioned. We would require this to train our model
- train.py: script to run our model. Trained model file as a output
- predict.py: script to run model inference. 


## Deployment

1. Clone the repository:
```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:
```
cd deploy/app
pip install -r requirements.txt
```

3. Build Docker image:
```
cd deploy/app
docker build -t your-image-name .
```

4. Deploy to Kubernetes:
```
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/service.yaml
```

## vertex_ai
- Sample codes for Google cloud deployment process
