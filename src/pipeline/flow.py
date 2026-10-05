"""Flujo de Prefect que conecta preparar -> entrenar -> comparar -> registrar.

Las etapas son marcadores de posición hasta definir los datos y los modelos.
"""
from prefect import flow, task, get_run_logger


@task
def prepare_data():
    get_run_logger().info("prepare_data: sin implementar")


@task
def train_models():
    get_run_logger().info("train_models: sin implementar")


@task
def compare_experiments():
    get_run_logger().info("compare_experiments: sin implementar")


@task
def register_model():
    get_run_logger().info("register_model: sin implementar")


@flow(name="defender-servidor")
def main_flow():
    prepare_data()
    train_models()
    compare_experiments()
    register_model()


if __name__ == "__main__":
    main_flow()
