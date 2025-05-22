# Orby Fork

## Installation
```
pip install .[vllm]
pip install -r orby/requirements.txt
boto3>=1.37.2
```

Note: we have to install separately because boto3 does not play nicely with s3fs, and we need both.
Don't worry about the incompatible version message (for now).
