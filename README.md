# vertex-ai-project-example
Welcome! This is an example project for demonstrating how to build an extensible, production-ready project on Vertex AI for image use cases.
It is built specifically for the [Mineral](https://x.company/projects/mineral/) team but has some ideas that might be useful to share.

## How to use
## Folder structure
- Data labelling
- Data processing
- Training
- Workspace

### Workspace
Here is where you create your datasets and training pipelines.

## Concepts
- Pipelines as extensible Python classes
- Automated training via Cloud Build

## Out-of-scope
MLOps, such as model monitoring, feature store, etc

# Setup

## Automation

### Cloud Build
Cloud Build packages your code and runs it on Google Cloud servers. 

Note: Instead of using `gcloud`, you can do all the below using the Google Cloud web console UI or even with client libraries.

1. Run all pipelines
The file to determine what commands are run are in the `.cloud-build/run_pipelines_cloudbuild.yaml` file.

To run this on the cloud, run the following command:
```
gcloud builds submit --config=.cloud-build/run_pipelines_cloudbuild.yaml --substitutions=_GCP_BUCKET_NAME="gs://MY_BUCKET_NAME/pipeline_staging"
```

This will error out as you haven't given the correct permissions to the service account yet.
To find out the service account, go the the Cloud Build logs listed in the terminal output for the command and find it under the "EXECUTION DETAILS" tab.

Alternatively, this command may give you the correct one:

```
gcloud projects get-iam-policy MY_PROJECT | grep cloudbuild
```


2. Find out what your Cloud Build service account is. You can try running the Cloud Build job once and seeing what account is used to execute it.

3. The Cloud Build service account needs access to the `aiplatform.pipelineJobs.create` permission. Grant it access to the `roles/aiplatform.user` role which has this permission.

```
gcloud projects add-iam-policy-binding MY_PROJECT \
      --member='serviceAccount:MY_SERVICE_ACCOUNT@cloudbuild.gserviceaccount.com' \
      --role='roles/aiplatform.user'
```

4. Also grant it the ability to use another service account by giving it the `roles/iam.serviceAccountUser` role:

```
gcloud projects add-iam-policy-binding MY_PROJECT \
      --member='serviceAccount:MY_SERVICE_ACCOUNT@cloudbuild.gserviceaccount.com' \
      --role='roles/iam.serviceAccountUser'
```

5. Now submit the build again. This time it should work.

```
gcloud builds submit --config=.cloud-build/run_pipelines_cloudbuild.yaml --substitutions=_GCP_BUCKET_NAME="gs://MY_BUCKET_NAME/pipeline_staging"
```

#### (Optional) Set up triggers
Instead of running the Cloud Build command from your local machine, you can set up triggers to automatically run your Cloud Build jobs.

For this repo, you might want to retrain all pipelines on the following conditions:
- Scheduled weekly
- Whenever someone creates a pull-request on your repositoy
- Whenever code is merged to a certain branch

Here are some examples:

1. Create a trigger to run pipelines whenever new code is merged to the 'main' branch:
```
gcloud beta builds triggers create github --name="run-pipelines" \
    --repo-name vertex-ai-project-example \
    --repo-owner ivanmkc \
    --branch-pattern="^main$" \
    --build-config .cloud-build/run_pipelines_cloudbuild.yaml
```

Also see this for more options: https://cloud.google.com/sdk/gcloud/reference/beta/builds/triggers/create/github
For more details, see: https://cloud.google.com/build/docs/automating-builds/build-repos-from-github

2. Create a trigger to run pipelines every week on Monday
There doesn't seem to be a way to do this without using the Google Cloud web console UI, so please see instructions here: https://cloud.google.com/build/docs/automating-builds/create-scheduled-triggers
