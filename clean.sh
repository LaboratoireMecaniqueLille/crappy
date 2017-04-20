#!/bin/sh
echo "This script will remove every compiled files, pyc and dist-packages"
echo "related to Crappy. RUN AT YOUR OWN RISK"
echo "This script requires trash-cli package and sudo access"
echo "Continue? (y/[n])"
read conf
if [ "$conf" != "y" ];then
  exit 0
fi
sudo trash-put ~/crappy/build ~/crappy/dist ~/crappy/crappy.egg-info
sudo trash-put /usr/local/lib/python2.7/dist-packages/crappy-*
find . -name "*~" -delete
n=`find . -name "*.pyc" |wc -l`
if [ $n = 0 ];then
  echo "No pyc files found, exiting..."
  exit 0
fi
find . -name "*.pyc"
echo "Delete these $n .pyc files? (y/[n])"
read conf
if [ "$conf" = "y" ];then
  find . -name "*.pyc" -delete
fi
