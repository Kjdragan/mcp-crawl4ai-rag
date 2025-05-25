# Google Cloud Run Python Client Library

## Overview

Google Cloud Run is a fully managed serverless platform that enables you to run stateless containers directly on top of Google's scalable infrastructure. The Google Cloud Run Python client library provides a Pythonic interface to the Cloud Run API, allowing developers to programmatically deploy, manage, and monitor their Cloud Run services.

## Key Features

- **Service Management**: Create, update, delete, and list Cloud Run services
- **Revision Management**: Manage service revisions and traffic splitting
- **Configuration**: Configure service settings like memory limits, CPU allocation, concurrency, and timeout
- **Autoscaling**: Control how your services scale automatically based on traffic
- **Domain Mapping**: Map custom domains to your Cloud Run services
- **IAM Integration**: Manage Identity and Access Management policies for your services
- **Monitoring**: Access metrics and logs for your deployed services

## Installation

```bash
# Install the Google Cloud Run client library
uv add google-cloud-run
```

## Authentication

Before using the Cloud Run client library, you need to set up authentication:

```python
# Set up authentication
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-key.json"
```

## Basic Usage Examples

### Initializing the Client

```python
from google.cloud import run_v2

# Initialize the services client
services_client = run_v2.ServicesClient()

# Initialize the revisions client
revisions_client = run_v2.RevisionsClient()

# Initialize the jobs client (for Cloud Run jobs)
jobs_client = run_v2.JobsClient()
```

### Listing Services

```python
from google.cloud import run_v2

def list_services(project_id, location="us-central1"):
    """List all Cloud Run services in a project and location."""
    client = run_v2.ServicesClient()
    
    # Format the parent resource name
    parent = f"projects/{project_id}/locations/{location}"
    
    # List services
    response = client.list_services(parent=parent)
    
    # Print service details
    for service in response:
        print(f"Service: {service.name}")
        print(f"  URL: {service.uri}")
        print(f"  Latest Revision: {service.latest_ready_revision}")
        print(f"  Traffic Allocations:")
        for traffic in service.traffic:
            print(f"    {traffic.percent}% -> {traffic.revision}")
        print()
    
    return response
```

### Creating a Service

```python
from google.cloud import run_v2
from google.cloud.run_v2.types import Service, RevisionTemplate, Container

def create_service(project_id, location, service_id, image_url):
    """Create a new Cloud Run service."""
    client = run_v2.ServicesClient()
    
    # Format the parent resource name
    parent = f"projects/{project_id}/locations/{location}"
    
    # Create the service configuration
    service = Service(
        template=RevisionTemplate(
            containers=[
                Container(
                    image=image_url,
                    resources={
                        "limits": {
                            "cpu": "1",
                            "memory": "512Mi"
                        }
                    },
                    ports=[{"container_port": 8080}]
                )
            ],
            scaling={
                "max_instance_count": 10,
                "min_instance_count": 0
            }
        ),
        labels={"environment": "production"}
    )
    
    # Create the service
    operation = client.create_service(
        parent=parent,
        service_id=service_id,
        service=service
    )
    
    # Wait for the operation to complete
    response = operation.result()
    
    print(f"Service {response.name} created successfully")
    print(f"Service URL: {response.uri}")
    
    return response
```

### Updating a Service

```python
from google.cloud import run_v2
from google.cloud.run_v2.types import Service, RevisionTemplate, Container

def update_service(project_id, location, service_id, new_image_url):
    """Update an existing Cloud Run service."""
    client = run_v2.ServicesClient()
    
    # Format the service name
    name = f"projects/{project_id}/locations/{location}/services/{service_id}"
    
    # Get the current service configuration
    service = client.get_service(name=name)
    
    # Update the container image
    service.template.containers[0].image = new_image_url
    
    # Update the service
    operation = client.update_service(service=service)
    
    # Wait for the operation to complete
    response = operation.result()
    
    print(f"Service {response.name} updated successfully")
    print(f"New revision: {response.latest_ready_revision}")
    
    return response
```

### Deleting a Service

```python
from google.cloud import run_v2

def delete_service(project_id, location, service_id):
    """Delete a Cloud Run service."""
    client = run_v2.ServicesClient()
    
    # Format the service name
    name = f"projects/{project_id}/locations/{location}/services/{service_id}"
    
    # Delete the service
    operation = client.delete_service(name=name)
    
    # Wait for the operation to complete
    operation.result()
    
    print(f"Service {name} deleted successfully")
```

### Working with Cloud Run Jobs

```python
from google.cloud import run_v2
from google.cloud.run_v2.types import Job, ExecutionTemplate, TaskTemplate, Container

def create_job(project_id, location, job_id, image_url):
    """Create a new Cloud Run job."""
    client = run_v2.JobsClient()
    
    # Format the parent resource name
    parent = f"projects/{project_id}/locations/{location}"
    
    # Create the job configuration
    job = Job(
        template=ExecutionTemplate(
            task_count=1,
            template=TaskTemplate(
                containers=[
                    Container(
                        image=image_url,
                        resources={
                            "limits": {
                                "cpu": "1",
                                "memory": "512Mi"
                            }
                        }
                    )
                ],
                max_retries=3,
                timeout="3600s"
            )
        ),
        labels={"environment": "production"}
    )
    
    # Create the job
    operation = client.create_job(
        parent=parent,
        job_id=job_id,
        job=job
    )
    
    # Wait for the operation to complete
    response = operation.result()
    
    print(f"Job {response.name} created successfully")
    
    return response
```

### Running a Job

```python
from google.cloud import run_v2

def run_job(project_id, location, job_id):
    """Run a Cloud Run job."""
    client = run_v2.JobsClient()
    
    # Format the job name
    name = f"projects/{project_id}/locations/{location}/jobs/{job_id}"
    
    # Run the job
    operation = client.run_job(name=name)
    
    # Wait for the operation to complete
    response = operation.result()
    
    print(f"Job execution {response.name} started")
    print(f"Status: {response.status}")
    
    return response
```

## Integration with Other Google Cloud Services

Cloud Run integrates seamlessly with other Google Cloud services:

- **Cloud Build**: For continuous deployment of your containers
- **Container Registry/Artifact Registry**: For storing your container images
- **Cloud Logging**: For viewing logs from your services
- **Cloud Monitoring**: For monitoring performance metrics
- **Secret Manager**: For securely accessing sensitive configuration
- **Cloud SQL**: For database connections
- **Pub/Sub**: For event-driven architectures

## Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Python Client Library Documentation](https://googleapis.dev/python/run/latest/index.html)
- [Google Cloud Run API Reference](https://cloud.google.com/run/docs/reference/rest)
- [GitHub Repository](https://github.com/googleapis/google-cloud-python/tree/main/packages/google-cloud-run)
