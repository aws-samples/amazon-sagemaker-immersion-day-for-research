# Deep Learning on AWS Open Data Registry: Automatic Building and Road Extraction from Satellite and LiDAR

This is a fork of the original tutorial at https://github.com/aws-samples/aws-open-data-satellite-lidar-tutorial.

## Setup

If you are running this as part of an AWS event, then these steps can be followed in the AWS Event engine provided account. Note that as part of the immersion day, we will only use the *-Lite notebooks.


### Create Conda environment
Next, set up a Conda environment by running `setup-env.sh` as shown below. You can change the environment name from `tutorial_env` to any other names.
```shell
$ ./setup-env.sh tutorial_env
```
This may take 10--15 minutes to complete.

Then check to make sure you have a new Jupyter kernel called `conda_tutorial_env`, or `conda_[name]` if you change the environment name to `[name]`. You may need to wait for a couple of minutes and refresh the Jupyter page.

### Download from S3 buckets
Next, download necessary files ([data browser](https://aws-satellite-lidar-tutorial.s3.amazonaws.com/index.html)) from S3 bucket prepared for this tutorial by running `download-from-s3.sh`:
```shell
$ ./download-from-s3.sh
```
This may take 5 minutes to complete, and requires at least 23GB of EBS disk size.

## Launch notebook
Finally, you can launch the notebooks `Building-Footprint-Lite.ipynb` or `Road-Network-Lite.ipynb` and learn to reproduce the tutorial. Note that if the notebook shows "No Kernel", or prompts to "Select Kernel", select the Jupyter kernel created in the previous step.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
The [NOTICE](THIRD-PARTY) includes third-party licenses used in this repository.

