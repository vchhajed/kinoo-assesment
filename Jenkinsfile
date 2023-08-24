pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                script {
                    // Build Docker image
                    dockerImage = docker.build("mlops-project:${BUILD_NUMBER}")
                }
            }
        }
        stage('Test') {
            steps {
                // Run tests here
            }
        }
        stage('Deploy') {
            steps {
                // Deploy using Kubernetes or other tools
            }
        }
    }
}
