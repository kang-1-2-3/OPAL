#!/bin/bash 
root=/data/Pcmaploc/data/Boreas
aws s3 sync s3://boreas/boreas-2020-12-01-13-26  $root/boreas-2020-12-01-13-26 --no-sign-request
aws s3 sync s3://boreas/boreas-2021-01-26-11-22  $root/boreas-2021-01-26-11-22 --no-sign-request
aws s3 sync s3://boreas/boreas-2021-04-29-15-55  $root/boreas-2021-04-29-15-55 --no-sign-request
aws s3 sync s3://boreas/boreas-2021-01-15-12-17  $root/boreas-2021-01-15-12-17 --no-sign-request
aws s3 sync s3://boreas/boreas-2021-03-02-13-38  $root/boreas-2021-03-02-13-38 --no-sign-request
aws s3 sync s3://boreas/boreas-2021-03-30-14-23  $root/boreas-2021-03-30-14-23 --no-sign-request
