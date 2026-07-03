# ============================================================
# DEPLOY MODEL TO SAGEMAKER ENDPOINT
# ============================================================

from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session
from sagemaker import get_execution_role

import sagemaker

# ============================================================
# CONFIGURATION
# ============================================================

session = Session()

role = get_execution_role()

bucket = session.default_bucket()

region = session.boto_region_name

print(f"Bucket : {bucket}")
print(f"Region : {region}")

# ============================================================
# MODEL ARTIFACT
# ============================================================

model_artifact = session.upload_data(
    path="models/model.tar.gz", bucket=bucket, key_prefix="credit-score-model"
)

print(model_artifact)

# ============================================================
# CREATE MODEL
# ============================================================

model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="sagemaker_inference.py",
    framework_version="1.7-1",
    py_version="py3",
    sagemaker_session=session,
)

# ============================================================
# DEPLOY
# ============================================================

predictor = model.deploy(
    endpoint_name="credit-score-endpoint",
    instance_type="ml.t3.medium",
    initial_instance_count=1,
)

print("Deployment Finished!")
