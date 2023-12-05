# Experiment on Batch Size vs Latency and Throughput
## Set up
- Create a new conda environment
```
$ conda create -n cs6220proj_bs_exp python=3.10
$ conda activate cs6220proj_bs_exp
```
- Install all necessary packages based on your device type
```
$ pip install -r requirement_gpu.txt
```

## How to run
- After generating benchmark data, specify the `LOG_DIRECTORY` (for the benchmark data) and `REPORT_DIRECTORY` (for the report) in `generate_report.sh`.
- Execute `generate_report.sh` to generate plots of the experiment.
```
$ ./generate_report.sh
```

## Reference
- AWS Reinvient 2021: https://github.com/aws-samples/aws-reinvent21-inf1-workshop