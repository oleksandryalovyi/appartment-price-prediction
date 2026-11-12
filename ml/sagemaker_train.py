"""Optional: the lab flow (stages 2, 4, 6) on SageMaker using the same encoded data.

Run inside SageMaker Studio / with AWS credentials:
    python ml/train.py --target rent --write-sagemaker-csv
    python ml/sagemaker_train.py --target rent            # trains + deploys a Serverless endpoint
Then run the backend with PREDICTOR=sagemaker SAGEMAKER_RENT_ENDPOINT=<printed name>.
Delete the endpoint when done:  python ml/sagemaker_train.py --delete <endpoint-name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import sagemaker
    from sagemaker import image_uris
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput
    from sagemaker.serializers import CSVSerializer
    from sagemaker.serverless import ServerlessInferenceConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["rent", "sale"])
    ap.add_argument("--role", help="SageMaker execution role ARN (defaults to the Studio role)")
    ap.add_argument("--delete", metavar="ENDPOINT")
    args = ap.parse_args()

    sess = sagemaker.Session()
    if args.delete:
        sess.delete_endpoint(args.delete)
        print("deleted", args.delete)
        return

    bucket, prefix = sess.default_bucket(), f"flat-price-xgb/{args.target}"
    local = ROOT / "ml" / "data" / "sagemaker" / args.target
    train_s3 = sess.upload_data(str(local / "train.csv"), bucket=bucket, key_prefix=f"{prefix}/train")
    val_s3 = sess.upload_data(str(local / "validation.csv"), bucket=bucket, key_prefix=f"{prefix}/validation")
    sess.upload_data(str(ROOT / "artifacts" / args.target / "columns.json"), bucket=bucket, key_prefix=f"{prefix}/meta")

    role = args.role or sagemaker.get_execution_role()
    image = image_uris.retrieve("xgboost", sess.boto_region_name, version="1.7-1")
    xgb = Estimator(image, role, instance_count=1, instance_type="ml.m5.large",
                    output_path=f"s3://{bucket}/{prefix}/output", sagemaker_session=sess)
    xgb.set_hyperparameters(objective="reg:squarederror", eval_metric="rmse", num_round=1500,
                            max_depth=7, eta=0.1, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=3, early_stopping_rounds=50)
    xgb.fit({"train": TrainingInput(train_s3, content_type="text/csv"),
             "validation": TrainingInput(val_s3, content_type="text/csv")})
    predictor = xgb.deploy(
        serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=5),
        serializer=CSVSerializer())
    print("ENDPOINT:", predictor.endpoint_name)


if __name__ == "__main__":
    main()
