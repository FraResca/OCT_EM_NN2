#!/bin/bash

sftp fresca@coka.fe.infn.it << EOF
cd OCT_EM_NN2
cd attention_maps
lcd attention_maps
# get *
cd ../results
lcd ../results
get *
cd ../saved_models
lcd ../saved_models
# get *
exit
EOF