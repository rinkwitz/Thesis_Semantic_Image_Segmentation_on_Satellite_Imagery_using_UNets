#!/bin/bash
aws s3 cp --recursive s3://spacenet-dataset/spacenet/SN1_buildings/test_public ./test_public
aws s3 cp --recursive s3://spacenet-dataset/spacenet/SN1_buildings/train ./train
