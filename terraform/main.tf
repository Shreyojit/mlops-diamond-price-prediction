provider "google" {
  credentials = file("gcp-key.json")
  project     = var.gcp_project_id
  region      = var.gcp_region
}

resource "google_cloud_run_service" "flask_app" {
  name     = "flask-app"
  location = var.gcp_region

  template {
    spec {
      containers {
        image = "docker.io/${var.dockerhub_username}/myflaskapp:latest"
        ports {
          container_port = 5000
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Grant public access to Cloud Run service
resource "google_cloud_run_service_iam_member" "public_access" {
  service   = google_cloud_run_service.flask_app.name
  location = var.gcp_region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "cloud_run_url" {
  value = google_cloud_run_service.flask_app.status[0].url
}
