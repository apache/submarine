-- Licensed to the Apache Software Foundation (ASF) under one or more
-- contributor license agreements.  See the NOTICE file distributed with
-- this work for additional information regarding copyright ownership.
-- The ASF licenses this file to You under the Apache License, Version 2.0
-- (the "License"); you may not use this file except in compliance with
-- the License.  You may obtain a copy of the License at
--    http://www.apache.org/licenses/LICENSE-2.0
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

CREATE TABLE if not exists experiment
(
    id                varchar(64) primary key,
    experiment_spec   text,
    create_by         varchar(32),
    create_time       datetime,
    update_by         varchar(32),
    update_time       datetime,
    experiment_status varchar(20),
    accepted_time     datetime,
    running_time      datetime,
    finished_time     datetime,
    uid               varchar(64)
    );

CREATE TABLE if not exists environment
(
    id               varchar(64) primary key,
    environment_name varchar(255),
    environment_spec text,
    create_by        varchar(32),
    create_time      datetime,
    update_by        varchar(32),
    update_time      datetime
    );
