# Commands for the lsv submit server

Check available compute resources:

````
watch -n 2 "condor_status -compact -af:h Machine NumDynamicSlots CPUs GPUs Memory State"
````

Check all running jobs:

````
condor_q -run -allusers
````
