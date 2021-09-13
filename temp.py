from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from ml_service.pipelines.load_sample_data import create_sample_data_csv
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os

e = Env()
# Get Azure machine learning workspace
aml_workspace = Workspace.get(
    name=e.workspace_name,
    subscription_id=e.subscription_id,
    resource_group=e.resource_group,
)
print("get_workspace:")
print(aml_workspace)

# Get Azure machine learning cluster
aml_compute = get_compute(aml_workspace, e.compute_name, e.vm_size)
if aml_compute is not None:
    print("aml_compute:")
    print(aml_compute)

# Create a reusable Azure ML environment
environment = get_environment(
    aml_workspace,
    e.aml_env_name,
    conda_dependencies_file=e.aml_env_train_conda_dep_file,
    create_new=e.rebuild_env,
)  #
run_config = RunConfiguration()
run_config.environment = environment

if e.datastore_name:
    datastore_name = e.datastore_name
else:
    datastore_name = aml_workspace.get_default_datastore().name
run_config.environment.environment_variables[
    "DATASTORE_NAME"
] = datastore_name  # NOQA: E501

model_name_param = PipelineParameter(name="model_name", default_value=e.model_name)  # NOQA: E501
dataset_version_param = PipelineParameter(
    name="dataset_version", default_value=e.dataset_version
)

caller_run_id_param = PipelineParameter(name="caller_run_id", default_value="none")  # NOQA: E501

# Get dataset name
dataset_name = e.dataset_name

# Check to see if dataset exists
if dataset_name not in aml_workspace.datasets:

    from azureml.core.dataset import Dataset
    web_paths = ['https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
                'https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
                'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
                'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz'
                ]
    dataset = Dataset.File.from_files(path = web_paths)

    # Register dataset
    dataset = dataset.register(workspace = aml_workspace,
                            name = 'mnist-dataset',
                            description='training and test dataset',
                            create_new_version=True)
