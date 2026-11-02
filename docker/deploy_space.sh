#!/bin/bash
# Upload the demo to a Hugging Face Space (Docker SDK). Needs HF_TOKEN, HF_SPACE (user/space), WEIGHTS_REPO.
# The Space needs its Dockerfile and a README with the Space settings at its root, so they are made here.
set -e
mkdir -p space
cp -r src app api requirements-demo.txt space/
cp docker/Dockerfile space/Dockerfile
echo "ENV WEIGHTS_REPO=$WEIGHTS_REPO" >> space/Dockerfile
cat > space/README.md <<END
---
title: Image to 3D Transformer
sdk: docker
app_port: 7860
---
Single-image 2D -> 3D reconstruction (airplane, car, chair). Code: https://github.com/iLogicus/image-to-3d-transformer
END
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$HF_SPACE', repo_type='space', space_sdk='docker', exist_ok=True)
api.upload_folder(folder_path='space', repo_id='$HF_SPACE', repo_type='space',
                  ignore_patterns=['**/__pycache__/**'], delete_patterns=['src/**', 'app/**', 'api/**'])
"
