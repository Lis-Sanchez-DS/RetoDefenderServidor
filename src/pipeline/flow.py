"""Prefect flow connecting prepare -> train -> compare -> register.

Stages are placeholders until data and models are decided.
"""
from prefect import flow, task, get_run_logger


@task
def prepare_data():
    get_run_logger().info("prepare_data: not implemented")


@task
def train_models():
    get_run_logger().info("train_models: not implemented")


@task
def compare_experiments():
    get_run_logger().info("compare_experiments: not implemented")


@task
def register_model():
    get_run_logger().info("register_model: not implemented")


@flow(name="defender-servidor")
def main_flow():
    prepare_data()
    train_models()
    compare_experiments()
    register_model()


if __name__ == "__main__":
    main_flow()
