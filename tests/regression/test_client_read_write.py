import contextlib
import os
import tempfile
import time

import pandas as pd
import pytest

from rubicon_ml import Rubicon


ARTIFACT_BINARY = b"artifact"
DATAFRAME = pd.DataFrame([[0]], columns=["column_a"])
TAGS_TO_ADD = ["added_tag_a", "added_tag_b"]
TAGS_TO_REMOVE = ["added_tag_a"]
COMMENTS_TO_ADD = ["added_comment_a", "added_comment_b"]
COMMENTS_TO_REMOVE = ["added_comment_a"]

CLIENTS_TO_TEST = [
    pytest.param("filesystem"),
    pytest.param("memory"),
    pytest.param("wandb", marks=pytest.mark.wandb),
]


def _test_read_write_additional_tags_and_comments(entity):
    """Test adding and removing tags and comments on an entity via client."""
    is_passing = True

    original_tags = list(entity.tags)
    entity.add_tags(TAGS_TO_ADD)
    entity.remove_tags(TAGS_TO_REMOVE)

    current_tags = list(entity.tags)
    expected_tags = original_tags + [t for t in TAGS_TO_ADD if t not in TAGS_TO_REMOVE]
    is_passing &= set(current_tags) == set(expected_tags)

    original_comments = list(entity.comments)
    entity.add_comments(COMMENTS_TO_ADD)
    entity.remove_comments(COMMENTS_TO_REMOVE)

    current_comments = list(entity.comments)
    expected_comments = original_comments + [c for c in COMMENTS_TO_ADD if c not in COMMENTS_TO_REMOVE]
    is_passing &= set(current_comments) == set(expected_comments)

    return is_passing


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_project_client_regression(project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the project entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_project_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)

        retrieved_project = rubicon.get_project(project.name)

        assert retrieved_project._domain == project._domain


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_experiment_client_regression(experiment_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the experiment entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_experiment_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)

        if persistence == "wandb":
            time.sleep(2)  # allow `wandb` time to complete sync

        retrieved_experiment = project.experiment(id=experiment.id)

        assert retrieved_experiment._domain == experiment._domain
        assert _test_read_write_additional_tags_and_comments(retrieved_experiment)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_feature_client_regression(feature_parameters, experiment_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the feature entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_feature_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)
        feature = experiment.log_feature(**feature_parameters)

        if persistence == "wandb":
            time.sleep(2)  # allow `wandb` time to complete sync

        retrieved_feature = experiment.feature(name=feature.name)

        assert retrieved_feature._domain == feature._domain
        assert _test_read_write_additional_tags_and_comments(retrieved_feature)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_metric_client_regression(metric_parameters, experiment_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the metric entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_metric_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)
        metric = experiment.log_metric(**metric_parameters)

        if persistence == "wandb":
            time.sleep(2)  # allow `wandb` time to complete sync

        retrieved_metric = experiment.metric(name=metric.name)

        assert retrieved_metric._domain == metric._domain
        assert _test_read_write_additional_tags_and_comments(retrieved_metric)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_parameter_client_regression(parameter_parameters, experiment_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the parameter entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_parameter_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)
        parameter = experiment.log_parameter(**parameter_parameters)

        if persistence == "wandb":
            time.sleep(2)  # allow `wandb` time to complete sync

        retrieved_parameter = experiment.parameter(name=parameter.name)

        assert retrieved_parameter._domain == parameter._domain
        assert _test_read_write_additional_tags_and_comments(retrieved_parameter)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_artifact_project_client_regression(artifact_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the artifact (project) entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_artifact_project_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        artifact = project.log_artifact(data_bytes=ARTIFACT_BINARY, **artifact_parameters)

        if persistence == "wandb":
            time.sleep(4)  # allow `wandb` time to complete sync

        retrieved_artifact = project.artifact(name=artifact.name)

        assert retrieved_artifact._domain == artifact._domain
        assert retrieved_artifact.get_data() == ARTIFACT_BINARY
        assert _test_read_write_additional_tags_and_comments(retrieved_artifact)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_artifact_experiment_client_regression(artifact_parameters, experiment_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the artifact (experiment) entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_artifact_experiment_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)
        artifact = experiment.log_artifact(data_bytes=ARTIFACT_BINARY, **artifact_parameters)

        if persistence == "wandb":
            time.sleep(4)  # allow `wandb` time to complete sync

        retrieved_artifact = experiment.artifact(name=artifact.name)

        assert retrieved_artifact._domain == artifact._domain
        assert retrieved_artifact.get_data() == ARTIFACT_BINARY
        assert _test_read_write_additional_tags_and_comments(retrieved_artifact)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_dataframe_project_client_regression(dataframe_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the dataframe (project) entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_dataframe_project_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        dataframe = project.log_dataframe(df=DATAFRAME, **dataframe_parameters)

        if persistence == "wandb":
            time.sleep(4)  # allow `wandb` time to complete sync

        retrieved_dataframe = project.dataframe(name=dataframe.name)

        assert retrieved_dataframe._domain == dataframe._domain
        assert retrieved_dataframe.get_data().equals(DATAFRAME)
        assert _test_read_write_additional_tags_and_comments(retrieved_dataframe)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_dataframe_experiment_client_regression(dataframe_parameters, experiment_parameters, project_parameters, persistence):
    """Tests that `rubicon_ml` client can read the dataframe (experiment) entity that it wrote."""
    if persistence == "filesystem":
        temp_dir_context = tempfile.TemporaryDirectory()
    else:
        temp_dir_context = contextlib.nullcontext(
            enter_result="./test_read_write_dataframe_experiment_client_regression/"
        )

    with temp_dir_context as temp_dir_name:
        root_dir = os.path.join(temp_dir_name, "test-rubicon-ml")
        rubicon = Rubicon(persistence=persistence, root_dir=root_dir)
        project = rubicon.create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)
        dataframe = experiment.log_dataframe(df=DATAFRAME, **dataframe_parameters)

        if persistence == "wandb":
            time.sleep(4)  # allow `wandb` time to complete sync

        retrieved_dataframe = experiment.dataframe(name=dataframe.name)

        assert retrieved_dataframe._domain == dataframe._domain
        assert retrieved_dataframe.get_data().equals(DATAFRAME)
        assert _test_read_write_additional_tags_and_comments(retrieved_dataframe)
