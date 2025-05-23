git config --global user.email "cheng@orby.ai"
git config --global user.name "Cheng Chang"

apt update
apt install -y emacs
apt install -y awscli
# urllib3<2 required by awscli
pip install 'urllib3<2'
pip install parquet-tools

# Download model.
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

# Install verl lib: https://verl.readthedocs.io/en/latest/start/install.html
pip3 install -e .[vllm]

# Download and convert uground dev set
# mkdir -p ~/data/uground/raw/
# aws s3 cp s3://orby-osu-va/mds_datasets/Q42024_Intake_Format/ActIO-ActionDescription/parquet/dev.parquet ~/data/uground/raw/dev.parquet
# python examples/data_preprocess/uground.py --input_file=~/data/uground/raw/dev.parquet --split=train
# python examples/data_preprocess/uground.py --input_file=~/data/uground/raw/dev.parquet --split=test

# Download the subtask direct distill dataset
mkdir -p ~/data/subtask_direct_distill/mix/train/
mkdir -p ~/data/subtask_direct_distill/mix/test/
aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/executor.parquet ~/data/subtask_direct_distill/mix/test/executor.parquet
aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/reward_model.parquet ~/data/subtask_direct_distill/mix/test/reward_model.parquet
aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/executor.parquet ~/data/subtask_direct_distill/mix/train/executor.parquet
aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/reward_model.parquet ~/data/subtask_direct_distill/mix/train/reward_model.parquet
