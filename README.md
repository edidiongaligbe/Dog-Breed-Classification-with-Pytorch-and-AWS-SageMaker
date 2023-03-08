# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I choose ResNet50 pretrained model.
The hyperparameters used are:
1. Learning rate - 0.001, 0.1.
2. Batch size - 32, 64, 128

Remember that your README should:
- Include a screenshot of completed training jobs
![Training_jobs.png](https://github.com/edidiongaligbe/Dog-Breed-Classification-with-Pytorch-and-AWS-SageMaker/blob/main/Training_jobs.PNG)
- Logs metrics during the training process
  Included in the train_and_deploy.ipynb file
- Tune at least two hyperparameters
  Batch size and Learning rate
- Retrieve the best best hyperparameters from all your training jobs
  Batch size: 32, Learning rate: 0.0036711776122006303

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
Installed smdebug, set ip debugger configurations, rules and profiler configurations. Set up the estimator with the configurations and initiated a training job.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
1. The batch size is too small.
2. GPUs are underutilized.
3. Initialization takes too long.

**TODO** Remember to provide the profiler html/pdf file in your submission.
- Included.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
import io
from PIL import Image
with open(filename, "rb") as f:
    payload = f.read()
    
type(payload)
#Run inference and output result
response=predictor.predict(payload, initial_args={"ContentType": "image/jpeg"})
response

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![endpoint](https://github.com/edidiongaligbe/Dog-Breed-Classification-with-Pytorch-and-AWS-SageMaker/blob/main/Endpoint.PNG)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
