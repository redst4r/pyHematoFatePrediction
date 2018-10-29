To host these notesbooks in the respective docker container (which has all the depencendies installed):
``` bash
docker run  -p8888:8888 -t -v /your/path/to/data_small:/mnt redst4r/pyhematoprediction --NotebookApp.iopub_data_rate_limit=1.0e10
# where redst4r/pyhematoprediction is the docker image
```
