# kinoo-assesment

## Installation

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