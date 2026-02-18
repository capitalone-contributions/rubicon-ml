import contextlib
import os
import tempfile

import pandas as pd
import pytest

from rubicon_ml import Rubicon
from rubicon_ml.exceptions import RubiconException

ARTIFACT_BINARY = b"artifact"
CLIENTS_TO_TEST = [
    pytest.param("filesystem"),
    pytest.param("memory"),
    pytest.param("wandb", marks=pytest.mark.wandb),
]
COMMENTS_TO_ADD = ["added comment a", "added comment b"]
COMMENTS_TO_REMOVE = ["added comment a"]
DATAFRAME = pd.DataFrame([[0]], columns=["column_a"])
EXPECTED_COMMENTS = ["comment a", "comment b", "added comment b"]
EXPECTED_TAGS = ["tag_a", "tag_b", "added_tag_b"]
TAGS_TO_ADD = ["added_tag_a", "added_tag_b"]
TAGS_TO_REMOVE = ["added_tag_a"]


def _domain_equals_excluding_tags_comments(domain_a, domain_b):
    dict_a = {k: v for k, v in domain_a.__dict__.items() if k not in ("tags", "comments")}
    dict_b = {k: v for k, v in domain_b.__dict__.items() if k not in ("tags", "comments")}

    return dict_a == dict_b


def _write_additional_tags_and_comments(entity):
    entity.add_tags(TAGS_TO_ADD)
    entity.remove_tags(TAGS_TO_REMOVE)
    entity.add_comments(COMMENTS_TO_ADD)
    entity.remove_comments(COMMENTS_TO_REMOVE)


def _test_read_write_additional_tags_and_comments(entity):
    return set(entity.tags) == set(EXPECTED_TAGS) and set(entity.comments) == set(EXPECTED_COMMENTS)


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
def test_read_write_experiment_client_regression(
    experiment_parameters, project_parameters, persistence
):
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

        _write_additional_tags_and_comments(experiment)

        if persistence == "wandb":
            rubicon.repository._active_run.finish()

        retrieved_experiment = project.experiment(id=experiment.id)

        assert _domain_equals_excluding_tags_comments(
            retrieved_experiment._domain, experiment._domain
        )
        assert _test_read_write_additional_tags_and_comments(retrieved_experiment)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
@pytest.mark.parametrize("is_existing_experiment", [True, False])
def test_read_write_feature_client_regression(
    feature_parameters,
    experiment_parameters,
    project_parameters,
    persistence,
    is_existing_experiment,
):
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
        project = rubicon.get_or_create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)

        if is_existing_experiment and persistence == "wandb":
            rubicon.repository._active_run.finish()
            experiment = project.experiment(id=experiment.id)

        feature = experiment.log_feature(**feature_parameters)
        _write_additional_tags_and_comments(feature)

        if persistence == "wandb":
            rubicon.repository._active_run.finish()

        retrieved_feature = experiment.feature(name=feature.name)

        assert _domain_equals_excluding_tags_comments(retrieved_feature._domain, feature._domain)
        assert _test_read_write_additional_tags_and_comments(retrieved_feature)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
@pytest.mark.parametrize("is_existing_experiment", [True, False])
def test_read_write_metric_client_regression(
    metric_parameters,
    experiment_parameters,
    project_parameters,
    persistence,
    is_existing_experiment,
):
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
        project = rubicon.get_or_create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)

        if is_existing_experiment and persistence == "wandb":
            rubicon.repository._active_run.finish()

            experiment = project.experiment(id=experiment.id)

        metric = experiment.log_metric(**metric_parameters)
        _write_additional_tags_and_comments(metric)

        if persistence == "wandb":
            rubicon.repository._active_run.finish()

        retrieved_metric = experiment.metric(name=metric.name)

        assert _domain_equals_excluding_tags_comments(retrieved_metric._domain, metric._domain)
        assert _test_read_write_additional_tags_and_comments(retrieved_metric)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
@pytest.mark.parametrize("is_existing_experiment", [True, False])
def test_read_write_parameter_client_regression(
    parameter_parameters,
    experiment_parameters,
    project_parameters,
    persistence,
    is_existing_experiment,
):
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
        project = rubicon.get_or_create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)

        if is_existing_experiment and persistence == "wandb":
            rubicon.repository._active_run.finish()
            experiment = project.experiment(id=experiment.id)

        parameter = experiment.log_parameter(**parameter_parameters)
        _write_additional_tags_and_comments(parameter)

        if persistence == "wandb":
            rubicon.repository._active_run.finish()

        retrieved_parameter = experiment.parameter(name=parameter.name)

        assert _domain_equals_excluding_tags_comments(
            retrieved_parameter._domain, parameter._domain
        )
        assert _test_read_write_additional_tags_and_comments(retrieved_parameter)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_artifact_project_client_regression(
    artifact_parameters, project_parameters, persistence
):
    """Tests that `rubicon_ml` client can read the artifact (project) entity that it wrote.

    For wandb persistence, this test verifies that project-level artifacts raise
    a RubiconException since W&B does not support project-level artifacts.
    """
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

        if persistence == "wandb":
            with pytest.raises(RubiconException, match="does not support project-level artifacts"):
                project.log_artifact(data_bytes=ARTIFACT_BINARY, **artifact_parameters)

            return

        artifact = project.log_artifact(data_bytes=ARTIFACT_BINARY, **artifact_parameters)

        _write_additional_tags_and_comments(artifact)

        retrieved_artifact = project.artifact(name=artifact.name)

        assert _domain_equals_excluding_tags_comments(retrieved_artifact._domain, artifact._domain)
        assert retrieved_artifact.get_data() == ARTIFACT_BINARY
        assert _test_read_write_additional_tags_and_comments(retrieved_artifact)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
@pytest.mark.parametrize("is_existing_experiment", [True, False])
def test_read_write_artifact_experiment_client_regression(
    artifact_parameters,
    experiment_parameters,
    project_parameters,
    persistence,
    is_existing_experiment,
):
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
        project = rubicon.get_or_create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)

        if is_existing_experiment and persistence == "wandb":
            rubicon.repository._active_run.finish()
            experiment = project.experiment(id=experiment.id)

        artifact = experiment.log_artifact(data_bytes=ARTIFACT_BINARY, **artifact_parameters)
        _write_additional_tags_and_comments(artifact)

        if persistence == "wandb":
            rubicon.repository._active_run.finish()

        retrieved_artifact = experiment.artifact(name=artifact.name)

        assert _domain_equals_excluding_tags_comments(retrieved_artifact._domain, artifact._domain)
        assert retrieved_artifact.get_data() == ARTIFACT_BINARY
        assert _test_read_write_additional_tags_and_comments(retrieved_artifact)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
def test_read_write_dataframe_project_client_regression(
    dataframe_parameters, project_parameters, persistence
):
    """Tests that `rubicon_ml` client can read the dataframe (project) entity that it wrote.

    For wandb persistence, this test verifies that project-level dataframes raise
    a RubiconException since W&B does not support project-level dataframes.
    """
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

        if persistence == "wandb":
            with pytest.raises(RubiconException, match="does not support project-level dataframes"):
                project.log_dataframe(df=DATAFRAME, **dataframe_parameters)

            return

        dataframe = project.log_dataframe(df=DATAFRAME, **dataframe_parameters)

        _write_additional_tags_and_comments(dataframe)

        retrieved_dataframe = project.dataframe(name=dataframe.name)

        assert _domain_equals_excluding_tags_comments(
            retrieved_dataframe._domain, dataframe._domain
        )
        assert retrieved_dataframe.get_data().equals(DATAFRAME)
        assert _test_read_write_additional_tags_and_comments(retrieved_dataframe)


@pytest.mark.parametrize("persistence", CLIENTS_TO_TEST)
@pytest.mark.parametrize("is_existing_experiment", [True, False])
def test_read_write_dataframe_experiment_client_regression(
    dataframe_parameters,
    experiment_parameters,
    project_parameters,
    persistence,
    is_existing_experiment,
):
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
        project = rubicon.get_or_create_project(**project_parameters)
        experiment = project.log_experiment(**experiment_parameters)

        if is_existing_experiment and persistence == "wandb":
            rubicon.repository._active_run.finish()
            experiment = project.experiment(id=experiment.id)

        dataframe = experiment.log_dataframe(df=DATAFRAME, **dataframe_parameters)
        _write_additional_tags_and_comments(dataframe)

        if persistence == "wandb":
            rubicon.repository._active_run.finish()

        retrieved_dataframe = experiment.dataframe(name=dataframe.name)

        assert _domain_equals_excluding_tags_comments(
            retrieved_dataframe._domain, dataframe._domain
        )
        assert retrieved_dataframe.get_data().equals(DATAFRAME)
        assert _test_read_write_additional_tags_and_comments(retrieved_dataframe)
