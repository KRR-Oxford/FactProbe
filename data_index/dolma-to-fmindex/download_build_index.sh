
# prepare dolma dataset 
# 1 and 2421 are the start and end gzip file numbers of the dolma dataset
# the dataset will be downloaed to ./data/raw
nohup bash ./helpers/download_dolma.sh 1 2421 > ./download_dolma/log_download.log 2>&1 &


# build fm-index 
# 1 and 2421 are the start and end gzip file numbers of the dolma dataset
# the text files extracted from the gzip files will be stored in ./data/processed
# the fm-index files will be stored in ./data/fm_index
# the script will use the sdsl-lite library to build the fm-index
nohup bash ./helpers/build_fmindex.sh 1 2421 > ./build_fmindex/log_build.log 2>&1 &
