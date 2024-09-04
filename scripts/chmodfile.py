import os
os.environ['KAGGLE_CONFIG_DIR'] = './' 

!chmod 600 ./kaggle.json

!chown `whoami`: ./kaggle.json

!kaggle datasets download -d mlg-ulb/creditcardfraud

!unzip \*.zip && rm *.zip