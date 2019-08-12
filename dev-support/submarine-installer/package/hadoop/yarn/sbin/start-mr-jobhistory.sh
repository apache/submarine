#!/bin/bash

cd `dirname $0`  
common_bin=`pwd`
YARN_LOGFILE=mr-jobhistory.log ${common_bin}/mr-jobhistory-daemon.sh start historyserver
