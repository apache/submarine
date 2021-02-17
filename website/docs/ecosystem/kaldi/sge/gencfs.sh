#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

WORKER_HOST_STRS=$1
SLOTS=$2

cat << EOF  > /tmp/qconf-mc.txt
arch                a           RESTRING    ==    YES         NO         NONE     0
calendar            c           RESTRING    ==    YES         NO         NONE     0
cpu                 cpu         DOUBLE      >=    YES         NO         0        0
display_win_gui     dwg         BOOL        ==    YES         NO         0        0
gpu                 g           INT         <=    YES         YES        0        10000
h_core              h_core      MEMORY      <=    YES         NO         0        0
h_cpu               h_cpu       TIME        <=    YES         NO         0:0:0    0
h_data              h_data      MEMORY      <=    YES         NO         0        0
h_fsize             h_fsize     MEMORY      <=    YES         NO         0        0
h_rss               h_rss       MEMORY      <=    YES         NO         0        0
h_rt                h_rt        TIME        <=    YES         NO         0:0:0    0
h_stack             h_stack     MEMORY      <=    YES         NO         0        0
h_vmem              h_vmem      MEMORY      <=    YES         NO         0        0
hostname            h           HOST        ==    YES         NO         NONE     0
load_avg            la          DOUBLE      >=    NO          NO         0        0
load_long           ll          DOUBLE      >=    NO          NO         0        0
load_medium         lm          DOUBLE      >=    NO          NO         0        0
load_short          ls          DOUBLE      >=    NO          NO         0        0
m_core              core        INT         <=    YES         NO         0        0
m_socket            socket      INT         <=    YES         NO         0        0
m_topology          topo        RESTRING    ==    YES         NO         NONE     0
m_topology_inuse    utopo       RESTRING    ==    YES         NO         NONE     0
mem_free            mf          MEMORY      <=    YES         NO         0        0
mem_total           mt          MEMORY      <=    YES         NO         0        0
mem_used            mu          MEMORY      >=    YES         NO         0        0
min_cpu_interval    mci         TIME        <=    NO          NO         0:0:0    0
np_load_avg         nla         DOUBLE      >=    NO          NO         0        0
np_load_long        nll         DOUBLE      >=    NO          NO         0        0
np_load_medium      nlm         DOUBLE      >=    NO          NO         0        0
np_load_short       nls         DOUBLE      >=    NO          NO         0        0
num_proc            p           INT         ==    YES         NO         0        0
qname               q           RESTRING    ==    YES         NO         NONE     0
ram_free            ram_free    MEMORY      <=    YES         JOB        0        0
rerun               re          BOOL        ==    NO          NO         0        0
s_core              s_core      MEMORY      <=    YES         NO         0        0
s_cpu               s_cpu       TIME        <=    YES         NO         0:0:0    0
s_data              s_data      MEMORY      <=    YES         NO         0        0
s_fsize             s_fsize     MEMORY      <=    YES         NO         0        0
s_rss               s_rss       MEMORY      <=    YES         NO         0        0
s_rt                s_rt        TIME        <=    YES         NO         0:0:0    0
s_stack             s_stack     MEMORY      <=    YES         NO         0        0
s_vmem              s_vmem      MEMORY      <=    YES         NO         0        0
seq_no              seq         INT         ==    NO          NO         0        0
slots               s           INT         <=    YES         YES        1        1000
swap_free           sf          MEMORY      <=    YES         NO         0        0
swap_rate           sr          MEMORY      >=    YES         NO         0        0
swap_rsvd           srsv        MEMORY      >=    YES         NO         0        0
swap_total          st          MEMORY      <=    YES         NO         0        0
swap_used           su          MEMORY      >=    YES         NO         0        0
tmpdir              tmp         RESTRING    ==    NO          NO         NONE     0
virtual_free        vf          MEMORY      <=    YES         NO         0        0
virtual_total       vt          MEMORY      <=    YES         NO         0        0
virtual_used        vu          MEMORY      >=    YES         NO         0        0
EOF

cat << EOF  > /tmp/qconf-ae.txt
hostname              $(hostname)
load_scaling          NONE
complex_values        ram_free=160G,gpu=2
user_lists            NONE
xuser_lists           NONE
projects              NONE
xprojects             NONE
usage_scaling         NONE
report_variables      NONE
EOF

cat << EOS  > /tmp/qconf-ap.txt
pe_name            smp
slots              32
user_lists         NONE
xuser_lists        NONE
start_proc_args    /bin/true
stop_proc_args     /bin/true
allocation_rule    \$pe_slots
control_slaves     FALSE
job_is_first_task  TRUE
urgency_slots      min
accounting_summary FALSE
EOS

cat << EOF > /tmp/qconf-aq.txt
qname                 all.q
hostlist              $WORKER_HOST_STRS
seq_no                0
load_thresholds       np_load_avg=1.75
suspend_thresholds    NONE
nsuspend              1
suspend_interval      00:05:00
priority              0
min_cpu_interval      00:05:00
processors            UNDEFINED
qtype                 BATCH INTERACTIVE
ckpt_list             NONE
pe_list               make smp
rerun                 FALSE
slots                 $SLOTS
tmpdir                /tmp
shell                 /bin/bash
prolog                NONE
epilog                NONE
shell_start_mode      posix_compliant
starter_method        NONE
suspend_method        NONE
resume_method         NONE
terminate_method      NONE
notify                00:00:60
owner_list            NONE
user_lists            NONE
xuser_lists           NONE
subordinate_list      NONE
complex_values        NONE
projects              NONE
xprojects             NONE
calendar              NONE
initial_state         default
s_rt                  INFINITY
h_rt                  INFINITY
s_cpu                 INFINITY
h_cpu                 INFINITY
s_fsize               INFINITY
h_fsize               INFINITY
s_data                INFINITY
h_data                INFINITY
s_stack               INFINITY
h_stack               INFINITY
s_core                INFINITY
h_core                INFINITY
s_rss                 INFINITY
h_rss                 INFINITY
s_vmem                INFINITY
h_vmem                INFINITY
EOF
