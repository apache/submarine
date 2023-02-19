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

CREATE TABLE `notebook`
(
    `id`              varchar(64) PRIMARY KEY,
    `notebook_spec`   text,
    `create_by`       varchar(32),
    `create_time`     datetime,
    `update_by`       varchar(32),
    `update_time`     datetime,
    `notebook_status` varchar(20),
    `notebook_url`    varchar(256),
    `reason`          varchar(512),
    `deleted_time`    datetime
);

insert into notebook (id, notebook_status)
values ('notebook_1642402491519_0003', 'starting');

CREATE TABLE `experiment`
(
    `id`                varchar(64) primary key,
    `experiment_spec`   text,
    `create_by`         varchar(32),
    `create_time`       datetime,
    `update_by`         varchar(32),
    `update_time`       datetime,
    `experiment_status` varchar(20),
    `accepted_time`     datetime,
    `running_time`      datetime,
    `finished_time`     datetime,
    `uid`               varchar(64)
);

insert into experiment (id, experiment_status)
values ('experiment-1659167632755-0001', 'Starting');
insert into experiment (id, experiment_status)
values ('experiment-1659167632755-0002', 'Starting');
insert into experiment (id, experiment_status)
values ('experiment-1659167632755-0003', 'Starting');
