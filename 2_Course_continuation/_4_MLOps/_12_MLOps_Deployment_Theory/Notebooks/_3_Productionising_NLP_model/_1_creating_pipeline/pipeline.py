"""Example workflow of pipeline script for complaints pipeline execution.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import logging
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

from botocore.exceptions import ClientError


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    """ Gets ECR URI from image versions
    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_versions: list of the image versions
        image_name: Name of the image

    Returns:
        ECR URI of the image version
    """

    #Fetch image details to get the Base Image URI
    for image_version in image_versions:
        if image_version['ImageVersionStatus'] == 'CREATED':
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info(f"Identified the latest image version: {image_arn}")
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name,
                Version=version
            )
            return response['ContainerImage']
    return None

def resolve_ecr_uri(sagemaker_session, image_arn):
    """Gets the ECR URI from the image name

    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_name: name of the image

    Returns:
        ECR URI of the latest image version
    """

    # Fetching image name from image_arn (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
    image_name = image_arn.partition("image/")[2]
    try:
        # Fetch the image versions
        next_token=''
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy='VERSION',
                SortOrder='DESCENDING',
                NextToken=next_token
            )
            ecr_uri = resolve_ecr_uri_from_image_versions(sagemaker_session, response['ImageVersions'], image_name)
            if "NextToken" in response:
                next_token = response["NextToken"]

            if ecr_uri is not None:
                return ecr_uri

        # Return error if no versions of the image found
        error_message = (
            f"No image version found for image name: {image_name}"
            )
        logger.error(error_message)
        raise Exception(error_message)

    except (ClientError, sagemaker_session.sagemaker_client.exceptions.ResourceNotFound) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="TicketClassificationNLP-p-ochbibbkoyxg",
    pipeline_name="CustomerComplaintsClassification",
    base_job_prefix="CustomerComplaints",
    project_id="p-ochbibbkoyxg",
    processing_instance_type="ml.m5.large",
    training_instance_type="ml.m5.large",
    inference_instance_type="ml.m5.large",
):
    """Gets a SageMaker ML Pipeline instance working with on complaints data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"https://sagemaker-project-p-2tz8o8pkejbu.s3.us-west-1.amazonaws.com/DataForModels/tfidffeatureswithlables.csv"    
    )
    processing_image_name = "sagemaker-{0}-processing-imagebuild".format(project_id)
    training_image_name = "sagemaker-{0}-training-imagebuild".format(project_id)
    inference_image_name = "sagemaker-{0}-inference-imagebuild".format(project_id)

    # processing step for feature engineering
    try:
        processing_image_uri = sagemaker_session.sagemaker_client.describe_image_version(ImageName=processing_image_name)['ContainerImage']
    except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
        processing_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.0-1",
            py_version="py3",
            instance_type=processing_instance_type,
        )
    script_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-complaints-preprocess",
        command=["python3"],
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )
    step_process = ProcessingStep(
        name="StepPreprocessComplaintsData",
        step_args=step_args,
    )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/ComplaintsTrain"

    try:
        training_image_uri = sagemaker_session.sagemaker_client.describe_image_version(ImageName=training_image_name)['ContainerImage']
    except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
        training_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.0-1",
            py_version="py3",
            instance_type=training_instance_type,
        )

    xgb_train = Estimator(
        image_uri=training_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/complaints-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        objective="multi:softmax",
        num_round=100,
        max_depth=8,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
        num_class=5,
    )
    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    step_train = TrainingStep(
        name="StepTrainComplaintsModel",
        step_args=step_args,
    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=training_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-complaint-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="ComplaintsEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="StepEvaluateComplaintsModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    try:
        inference_image_uri = sagemaker_session.sagemaker_client.describe_image_version(ImageName=inference_image_name)['ContainerImage']
    except (sagemaker_session.sagemaker_client.exceptions.ResourceNotFound):
        inference_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.0-1",
            py_version="py3",
            instance_type=inference_instance_type,
        )
    model = Model(
        image_uri=inference_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name="StepRegisterComplaintsModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value"
        ),
        right=1,
    )
    step_cond = ConditionStep(
        name="CheckAccuracyComplaintEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline