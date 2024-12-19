# Create a new task
curl -X POST http://localhost:7070/api/v1/image-to-3d \
  -H 'Content-Type: application/json' \
  -d '{
    "image_url": "https://raw.githubusercontent.com/microsoft/TRELLIS/refs/heads/main/assets/example_image/T.png",
    "segm_mode": "auto"
  }'

# Response should be something like:
# {"id": "123e4567-e89b-12d3-a456-426614174000"}

# Then check the status
curl http://localhost:7070/api/v1/image-to-3d/123e4567-e89b-12d3-a456-426614174000