import mlflow
import ingest_data
import train
# import score

# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000
remote_server_uri = "http://tiger0219.tigeranalytics.local:5000" # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

exp_name = "MLE_Training9"
mlflow.set_experiment(exp_name)

# Create nested runs
with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
    mlflow.log_param("parent", "yes")

    with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
        ingest_data.main()
        mlflow.log_param("data download", "yes")

    with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
        train.main()
        mlflow.log_param("train", "yes")

    # with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
    #     score.main()
    #     mlflow.log_param("score", "yes")

print("parent run_id: {}".format(parent_run.info.run_id))