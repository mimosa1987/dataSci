import os

PROJECT_PATH = os.getcwd()
config_dir = os.path.join(PROJECT_PATH, "conf")
log_dir = os.path.join(PROJECT_PATH, "log")
data_dir = os.path.join(PROJECT_PATH, "data")
global_config = {
    "project_path": PROJECT_PATH,
    "config_dir" : config_dir,
    "log_dir" : log_dir,
    "data_dir" : data_dir,
    "db_config": os.path.join(config_dir, "db_config.ini"),
    "encoder": os.path.join(config_dir, "encoder_config.json"),
    "model": os.path.join(config_dir, "model_config.json"),
    "strategy": os.path.join(config_dir, "strategy_config.json"),
    "feature": os.path.join(config_dir, "feature_config.json"),
    "job": os.path.join(config_dir, "jobs_config.json"),
    "workflow": os.path.join(config_dir, "workflow.json")
}
