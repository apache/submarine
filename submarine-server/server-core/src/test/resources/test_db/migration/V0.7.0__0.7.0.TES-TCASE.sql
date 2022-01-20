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

-- ----------------------------
-- Test table structure for sys_dict
-- ----------------------------
CREATE TABLE IF NOT EXISTS sys_user
(
    id          varchar(32)  NOT NULL COMMENT 'id' primary key ,
    user_name   varchar(32)  NOT NULL COMMENT 'login name',
    real_name   varchar(64)  NOT NULL COMMENT 'real name',
    password    varchar(255) NOT NULL COMMENT 'password',
    avatar      varchar(255) default NULL COMMENT 'avatar',
    birthday    datetime     default NULL COMMENT 'birthday',
    sex         varchar(32)  default NULL COMMENT 'sex',
    email       varchar(32)  default NULL COMMENT 'email',
    phone       varchar(32)  default NULL COMMENT 'telphone',
    dept_code   varchar(32)  default NULL COMMENT 'department code',
    role_code   varchar(32)  default NULL COMMENT 'role code',
    status      varchar(32)  default NULL COMMENT 'status',
    deleted     int(1)       default 0 COMMENT 'deleted status(0:normal, 1:already deleted)',
    create_by   varchar(32)  default NULL COMMENT 'create user',
    create_time datetime     default NULL COMMENT 'create time',
    update_by   varchar(32)  default NULL COMMENT 'last update user',
    update_time datetime     default NULL COMMENT 'last update time'
);

-- ----------------------------
-- Test records of sys_user
-- ----------------------------
INSERT INTO sys_user VALUES ('e9ca23d68d884d4ebb19d07889727dae', 'admin', 'administrator', '21232f297a57a5a743894a0e4a801fc3', 'avatar.png', '2018-12-05 00:00:00', NULL, 'dev@submarine.org', '18566666661', NULL, NULL, NULL, 0, 'admin', '2019-07-05 14:47:22', 'admin', '2019-07-05 14:47:22');
