#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/usr/bin/env sh
###########################################################################################################
# NAME: stopTonyPortal.sh
#
# DESCRIPTION:
# This script stops the Tony Portal process.
#
#
# INPUT:
# $1 - path of RUNNING_PID of TonY Portal (Optional. Default to current folder)
#
#
# EXIT CODE:
# 0 - Success
# 1 - Failed to find RUNNING_PID
#
#
# CHANGELOG:
# DEC 10 2018 PHAT TRAN
############################################################################################################
RUNNING_PID_PATH=./RUNNING_PID
if [ ! -z "$1" ]; then
    RUNNING_PID_PATH=$1
fi

PID=`cat $RUNNING_PID_PATH`
if [ $? -ne 0 ]; then
    echo "Invalid path to RUNNING_PID"
    exit 1
fi

kill -9 $PID

rm $RUNNING_PID_PATH # So we can run startTonyPortal.sh again
