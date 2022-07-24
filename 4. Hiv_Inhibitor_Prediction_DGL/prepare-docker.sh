#!/bin/bash

sudo service docker stop
sudo cp -rp /var/lib/docker /home/ec2-user/SageMaker/docker
sudo service docker start