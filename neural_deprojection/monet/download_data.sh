#!/usr/env bash

kaggle competitions download -c gan-getting-started
mkdir data
mv gan-getting-started.zip data
cd data
unzip gan-getting-started.zip
cd ..
