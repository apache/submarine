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
-- Table structure for sys_dict
-- ----------------------------
CREATE TABLE IF NOT EXISTS `sys_dict` (
  `id` varchar(32) NOT NULL,
  `dict_code` varchar(32) NOT NULL COMMENT 'dict code',
  `dict_name` varchar(32) NOT NULL COMMENT 'dict name',
  `description` varchar(255) default NULL COMMENT 'dict description',
  `deleted` int(1) default 0 COMMENT 'delete status(0:normal, 1:already deleted)',
  `type` int(1) default 0 COMMENT 'dict type (0:string,1:number)',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `UK_SYS_DICT_DICT_CODE` (`dict_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for sys_dict_item
-- ----------------------------
CREATE TABLE IF NOT EXISTS `sys_dict_item` (
  `id` varchar(32) NOT NULL,
  `item_code` varchar(32) NOT NULL COMMENT 'dict item code',
  `item_name` varchar(32) NOT NULL COMMENT 'dict item name',
  `dict_code` varchar(32) NOT NULL COMMENT 'dict code',
  `description` varchar(255) default NULL COMMENT 'description',
  `sort_order` int(3) default 0 COMMENT 'sort order',
  `deleted` int(1) default 0 COMMENT 'delete status(0:normal,1:already deleted)',
  `create_by` varchar(32) default NULL,
  `create_time` datetime default NULL,
  `update_by` varchar(32) default NULL,
  `update_time` datetime default NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `UK_SYS_DICT_ITEM_CODE` (`item_code`)/*,
  CONSTRAINT `FK_SYS_DICT_ITEM_DICT_CODE` FOREIGN KEY (`dict_code`) REFERENCES `sys_dict` (`dict_code`)*/
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for system department
-- ----------------------------
CREATE TABLE IF NOT EXISTS  `sys_department` (
  `id` varchar(32) NOT NULL COMMENT 'ID',
  `dept_code` varchar(32) NOT NULL COMMENT 'department code',
  `dept_name` varchar(64) NOT NULL COMMENT 'department name',
  `parent_code` varchar(32) default NULL COMMENT 'parent dept code',
  `sort_order` int(3) default 0 COMMENT 'sort order',
  `description` varchar(255) COMMENT 'description',
  `deleted` varchar(1) default 0 COMMENT 'delete status(0:normal,1:already deleted)',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `UK_DEPT_CODE` (`dept_code`)/*,
  CONSTRAINT `FK_SYS_DEPT_PARENT_CODE` FOREIGN KEY (`parent_code`) REFERENCES `sys_department` (`dept_code`)*/
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for sys_user
-- ----------------------------
CREATE TABLE IF NOT EXISTS `sys_user` (
  `id` varchar(32) NOT NULL COMMENT 'id',
  `user_name` varchar(32) NOT NULL COMMENT 'login name',
  `real_name` varchar(64) NOT NULL COMMENT 'real name',
  `password` varchar(255) NOT NULL COMMENT 'password',
  `avatar` varchar(255) default NULL COMMENT 'avatar',
  `birthday` datetime default NULL COMMENT 'birthday',
  `sex` varchar(32) default NULL COMMENT 'sex',
  `email` varchar(32) default NULL COMMENT 'email',
  `phone` varchar(32) default NULL COMMENT 'telphone',
  `dept_code` varchar(32) default NULL COMMENT 'department code',
  `role_code` varchar(32) default NULL COMMENT 'role code',
  `status` varchar(32) default NULL COMMENT 'status',
  `deleted` int(1) default 0 COMMENT 'deleted status(0:normal, 1:already deleted)',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `sys_user_name` (`user_name`)/*,
  CONSTRAINT `FK_SYS_USER_DEPT_CODE` FOREIGN KEY (`dept_code`) REFERENCES `sys_department` (`dept_code`),
  CONSTRAINT `FK_SYS_USER_SEX` FOREIGN KEY (`sex`) REFERENCES `sys_dict_item` (`item_code`),
  CONSTRAINT `FK_SYS_USER_STATUS` FOREIGN KEY (`status`) REFERENCES `sys_dict_item` (`item_code`)*/
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for sys_message
-- ----------------------------
CREATE TABLE IF NOT EXISTS  `sys_message` (
  `id` varchar(32) NOT NULL COMMENT 'id',
  `sender` varchar(32) default NULL COMMENT 'sender user',
  `receiver` varchar(32) default NULL COMMENT 'receiver user',
  `type` varchar(32) default NULL COMMENT 'dict_code:MESSAGE_TYPE',
  `context` text COMMENT 'message context',
  `status` int(1) default 0 COMMENT '0:unread, 1:read',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`)/*,
  CONSTRAINT `FK_SYS_MSG_SENDER` FOREIGN KEY (`sender`) REFERENCES `sys_user` (`user_name`),
  CONSTRAINT `FK_SYS_MSG_RECEIVER` FOREIGN KEY (`receiver`) REFERENCES `sys_user` (`user_name`),
  CONSTRAINT `FK_SYS_MSG_TYPE` FOREIGN KEY (`type`) REFERENCES `sys_dict_item` (`item_code`)*/
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for team
-- ----------------------------
CREATE TABLE IF NOT EXISTS  `team` (
  `id` varchar(32) NOT NULL,
  `owner` varchar(100) NOT NULL COMMENT 'owner name',
  `team_name` varchar(64) NOT NULL COMMENT 'team name',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `UK_TEAM_NAME` (`team_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for team_member
-- ----------------------------
CREATE TABLE IF NOT EXISTS `team_member` (
  `id` varchar(32) NOT NULL,
  `team_id` varchar(32) NOT NULL COMMENT 'team id',
  `team_name` varchar(64) NOT NULL COMMENT 'team name',
  `member` varchar(100) NOT NULL COMMENT 'member name',
  `inviter` int(1) default 0 COMMENT '0:inviter, 1:accept',
  `create_by` varchar(32) default NULL,
  `create_time` datetime default NULL,
  `update_by` varchar(32) default NULL,
  `update_time` datetime default NULL,
  PRIMARY KEY  (`id`)/*,
  CONSTRAINT `FK_TEAM_MEMBER_USER` FOREIGN KEY (`member`) REFERENCES `sys_user` (`user_name`)*/
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for project
-- ----------------------------
CREATE TABLE IF NOT EXISTS `project` (
  `id` varchar(32) NOT NULL,
  `name` varchar(100) NOT NULL COMMENT 'project name',
  `visibility` varchar(32) default NULL COMMENT 'dict_code:PROJECT_VISIBILITY',
  `permission` varchar(32) default NULL COMMENT 'dict_code:PROJECT_PERMISSION',
  `type` varchar(32) default NULL COMMENT 'dict_code:PROJECT_TYPE',
  `description` varchar(255) COMMENT 'description',
  `user_name` varchar(32) NOT NULL COMMENT 'owner user id',
  `team_name` varchar(32) default NULL COMMENT 'team name',
  `tags` varchar(128) default NULL COMMENT 'Comma separated tag',
  `star_num` int(8) default 0 COMMENT 'star number',
  `like_num` int(8) default 0 COMMENT 'like number',
  `message_num` int(8) default 0 COMMENT 'message number',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`)/*,
  CONSTRAINT `FK_PROJECT_TYPE` FOREIGN KEY (`type`) REFERENCES `sys_dict_item` (`item_code`),
  CONSTRAINT `FK_PROJECT_TEAM_NAME` FOREIGN KEY (`team_name`) REFERENCES `team` (`team_name`),
  CONSTRAINT `FK_PROJECT_USER_NAME` FOREIGN KEY (`user_name`) REFERENCES `sys_user` (`user_name`)*/
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for project_files
-- ----------------------------
CREATE TABLE IF NOT EXISTS `project_files` (
  `id` varchar(32) NOT NULL,
  `project_id` varchar(32) NOT NULL COMMENT 'project id',
  `file_name` varchar(128) NOT NULL COMMENT '/path/.../file.suffix',
  `file_content` text default NULL COMMENT 'file content',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`)/*,
  CONSTRAINT `FK_PROJECT_FILES_PRJ_ID` FOREIGN KEY (`project_id`) REFERENCES `project` (`id`)*/
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for jobs
-- ----------------------------
CREATE TABLE IF NOT EXISTS `job` (
  `id` int NOT NULL AUTO_INCREMENT,
  `job_id` varchar(64) default NULL COMMENT 'job id',
  `job_name` varchar(64) NOT NULL COMMENT 'job name',
  `job_type` varchar(64) NOT NULL COMMENT 'job type',
  `job_namespace` varchar(32) default NULL COMMENT 'job namespace',
  `job_status` varchar(32) default NULL COMMENT 'job status',
  `job_final_status` varchar(32) default NULL COMMENT 'job final status',
  `user_name` varchar(32) default NULL COMMENT 'user name',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for environment
-- ----------------------------
CREATE TABLE IF NOT EXISTS `environment` (
  `id` varchar(64) NOT NULL COMMENT 'Id of the Environment',
  `environment_name` varchar(255) NOT NULL COMMENT 'Name of the Environment',
  `environment_spec` text NOT NULL COMMENT 'Spec of the Environment',
  `create_by` varchar(32) DEFAULT NULL COMMENT 'create user',
  `create_time` datetime DEFAULT NULL COMMENT 'create time',
  `update_by` varchar(32) DEFAULT NULL COMMENT 'last update user',
  `update_time` datetime DEFAULT NULL COMMENT 'last update time',
   PRIMARY KEY `id` (`id`),
   UNIQUE KEY `environment_name` (`environment_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for experiment
-- ----------------------------
CREATE TABLE IF NOT EXISTS `experiment` (
  `id` varchar(64) NOT NULL COMMENT 'Id of the Experiment',
  `experiment_spec` text NOT NULL COMMENT 'Spec of the experiment',
  `create_by` varchar(32) DEFAULT NULL COMMENT 'create user',
  `create_time` datetime DEFAULT NULL COMMENT 'create time',
  `update_by` varchar(32) DEFAULT NULL COMMENT 'last update user',
  `update_time` datetime DEFAULT NULL COMMENT 'last update time',
  `experiment_status` varchar(20) DEFAULT NULL COMMENT 'experiment status',
   PRIMARY KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for notebook
-- ----------------------------
CREATE TABLE IF NOT EXISTS `notebook` (
  `id` varchar(64) NOT NULL COMMENT 'Id of the notebook',
  `notebook_spec` text NOT NULL COMMENT 'Spec of the notebook',
  `create_by` varchar(32) DEFAULT NULL COMMENT 'create user',
  `create_time` datetime DEFAULT NULL COMMENT 'create time',
  `update_by` varchar(32) DEFAULT NULL COMMENT 'last update user',
  `update_time` datetime DEFAULT NULL COMMENT 'last update time',
  `notebook_status` varchar(20) DEFAULT NULL COMMENT 'notebook status',
   PRIMARY KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for experiment_templates
-- ----------------------------
CREATE TABLE IF NOT EXISTS `experiment_template` (
  `id` varchar(64) NOT NULL,
  `experimentTemplate_name` varchar(32) NOT NULL,
  `experimentTemplate_spec` json DEFAULT NULL,
  `create_by` varchar(32) DEFAULT NULL,
  `create_time` datetime NOT NULL,
  `update_by` varchar(32) DEFAULT NULL,
  `update_time` datetime NOT NULL,
  PRIMARY KEY `id` (`id`),
   UNIQUE KEY `experimentTemplate_name` (`experimentTemplate_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for registered_model
-- ----------------------------
CREATE TABLE IF NOT EXISTS `registered_model` (
  `name` VARCHAR(256) NOT NULL,
  `creation_time` DATETIME(3) COMMENT 'Millisecond precision',
  `last_updated_time` DATETIME(3) COMMENT 'Millisecond precision',
  `description` VARCHAR(5000),
  CONSTRAINT `registered_model_pk` PRIMARY KEY (`name`),
  UNIQUE (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for registered_model
-- ----------------------------
CREATE TABLE IF NOT EXISTS `registered_model_tag` (
  `name` VARCHAR(256) NOT NULL,
  `tag` VARCHAR(256) NOT NULL,
  CONSTRAINT `registered_model_tag_pk` PRIMARY KEY (`name`, `tag`),
  FOREIGN KEY(`name`) REFERENCES `registered_model` (`name`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for model_version
-- ----------------------------
CREATE TABLE IF NOT EXISTS `model_version` (
  `name` VARCHAR(256) NOT NULL COMMENT 'Name of model',
  `version` INTEGER NOT NULL,
  `source` VARCHAR(512) NOT NULL COMMENT 'Model saved link',
  `user_id` VARCHAR(64) NOT NULL COMMENT 'Id of the created user',
  `experiment_id` VARCHAR(64) NOT NULL,
  `model_type` VARCHAR(64) NOT NULL COMMENT 'Type of model',
  `current_stage` VARCHAR(64) COMMENT 'Model stage ex: None, production...',
  `creation_time` DATETIME(3) COMMENT 'Millisecond precision',
  `last_updated_time` DATETIME(3) COMMENT 'Millisecond precision',
  `dataset` VARCHAR(256) COMMENT 'Which dataset is used',
  `description` VARCHAR(5000),
  CONSTRAINT `model_version_pk` PRIMARY KEY (`name`, `version`),
  FOREIGN KEY(`name`) REFERENCES `registered_model` (`name`) ON UPDATE CASCADE ON DELETE CASCADE,
  UNIQUE(`source`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for model_version_tag
-- ----------------------------
CREATE TABLE IF NOT EXISTS `model_version_tag` (
  `name` VARCHAR(256) NOT NULL COMMENT 'Name of model',
  `version` INTEGER NOT NULL,
  `tag` VARCHAR(256) NOT NULL,
  CONSTRAINT `model_version_tag_pk` PRIMARY KEY (`name`, `version`, `tag`),
  FOREIGN KEY(`name`, `version`) REFERENCES `model_version` (`name`, `version`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for metric
-- ----------------------------
CREATE TABLE IF NOT EXISTS `metric` (
  `id` VARCHAR(64) NOT NULL COMMENT 'Id of the Experiment',
  `key` VARCHAR(190) NOT NULL COMMENT 'Metric key: `String` (limit 190 characters). Part of *Primary Key* for ``metric`` table.',
  `value` FLOAT NOT NULL COMMENT 'Metric value: `Float`. Defined as *Non-null* in schema.',
  `worker_index` VARCHAR(32) NOT NULL COMMENT 'Metric worker_index: `String` (limit 32 characters). Part of *Primary Key* for\r\n    ``metrics`` table.',
  `timestamp` DATETIME(3) NOT NULL COMMENT 'Timestamp recorded for this metric entry: `DATETIME` (millisecond precision).
                       Part of *Primary Key* for   ``metrics`` table.',
  `step` INTEGER NOT NULL COMMENT 'Step recorded for this metric entry: `INTEGER`.',
  `is_nan` BOOLEAN NOT NULL COMMENT 'True if the value is in fact NaN.',
  CONSTRAINT `metric_pk` PRIMARY KEY  (`id`, `key`, `timestamp`, `worker_index`),
  FOREIGN KEY(`id`) REFERENCES `experiment` (`id`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for param
-- ----------------------------
CREATE TABLE IF NOT EXISTS `param` (
  `id` VARCHAR(64) NOT NULL COMMENT 'Id of the Experiment',
  `key` VARCHAR(190) NOT NULL COMMENT '`String` (limit 190 characters). Part of *Primary Key* for ``param`` table.',
  `value` VARCHAR(190) NOT NULL COMMENT '`String` (limit 190 characters). Defined as *Non-null* in schema.',
  `worker_index` VARCHAR(32) NOT NULL COMMENT '`String` (limit 32 characters). Part of *Primary Key* for\r\n    ``metric`` table.',
  CONSTRAINT `param_pk` PRIMARY KEY  (`id`, `key`, `worker_index`),
  FOREIGN KEY(`id`) REFERENCES `experiment` (`id`) ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of sys_dict
-- ----------------------------
INSERT INTO `sys_dict` VALUES ('ca2dd544ca4c11e9a71e0242ac110002','SYS_USER_SEX','Sys user sex','submarine system dict, Do not modify.',0,0,NULL,'2019-08-29 11:04:36',NULL,'2019-09-01 01:08:12');
INSERT INTO `sys_dict` VALUES ('f405a7b1cc5411e9af810242ac110002','SYS_USER_STATUS','Sys user status','submarine system dict, Do not modify.',0,0,NULL,'2019-09-01 01:08:05',NULL,'2019-09-01 01:08:05');
INSERT INTO `sys_dict` VALUES ('3a1ed33ae83611e9ab840242ac110002','PROJECT_TYPE','Project machine learning type','submarine system dict, Do not modify.',0,0,NULL,'2019-09-01 01:08:05',NULL,'2019-09-01 01:08:05');
INSERT INTO `sys_dict` VALUES ('8a101495e84011e9ab840242ac110002','PROJECT_VISIBILITY','Project visibility type','submarine system dict, Do not modify.',0,0,NULL,'2019-09-01 01:08:05',NULL,'2019-09-01 01:08:05');
INSERT INTO `sys_dict` VALUES ('8f0439c9e84011e9ab840242ac110002','PROJECT_PERMISSION','Project permission type','submarine system dict, Do not modify.',0,0,NULL,'2019-09-01 01:08:05',NULL,'2019-09-01 01:08:05');

-- ----------------------------
-- Records of sys_dict_item
-- ----------------------------
INSERT INTO `sys_dict_item` VALUES ('27ef1080cc5511e9af810242ac110002','SYS_USER_STATUS_AVAILABLE','Available','SYS_USER_STATUS','submarine system dict, Do not modify.',1,0,NULL,'2019-09-01 01:09:32',NULL,'2019-09-01 01:13:19');
INSERT INTO `sys_dict_item` VALUES ('4c5d736acc5511e9af810242ac110002','SYS_USER_STATUS_LOCKED','Locked','SYS_USER_STATUS','submarine system dict, Do not modify.',2,0,NULL,'2019-09-01 01:10:33',NULL,'2019-09-01 01:12:53');
INSERT INTO `sys_dict_item` VALUES ('6d5ae3b2cc5511e9af810242ac110002','SYS_USER_STATUS_REGISTERED','New Registered','SYS_USER_STATUS','submarine system dict, Do not modify.',3,0,NULL,'2019-09-01 01:11:29',NULL,'2019-09-01 01:12:47');
INSERT INTO `sys_dict_item` VALUES ('d018e2b0ca4c11e9a71e0242ac110002','SYS_USER_SEX_MALE','Male','SYS_USER_SEX','submarine system dict, Do not modify.',1,0,NULL,'2019-08-29 11:04:46',NULL,'2019-09-01 00:53:54');
INSERT INTO `sys_dict_item` VALUES ('d94410adca4c11e9a71e0242ac110002','SYS_USER_SEX_FEMALE','Female','SYS_USER_SEX','submarine system dict, Do not modify.',2,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');
INSERT INTO `sys_dict_item` VALUES ('7b9aafa0e83611e9ab840242ac110002','PROJECT_TYPE_NOTEBOOK','notebook','PROJECT_TYPE','submarine system dict, Do not modify.',1,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');
INSERT INTO `sys_dict_item` VALUES ('8229a76be83611e9ab840242ac110002','PROJECT_TYPE_PYTHON','python','PROJECT_TYPE','submarine system dict, Do not modify.',2,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');
INSERT INTO `sys_dict_item` VALUES ('ac80ab12e83611e9ab840242ac110002','PROJECT_TYPE_R','R','PROJECT_TYPE','submarine system dict, Do not modify.',3,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');
INSERT INTO `sys_dict_item` VALUES ('b1070158e83611e9ab840242ac110002','PROJECT_TYPE_SCALA','scala','PROJECT_TYPE','submarine system dict, Do not modify.',4,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');
INSERT INTO `sys_dict_item` VALUES ('8c53870be83611e9ab840242ac110002','PROJECT_TYPE_TENSORFLOW','tensorflow','PROJECT_TYPE','submarine system dict, Do not modify.',5,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');
INSERT INTO `sys_dict_item` VALUES ('90ca63dfe83611e9ab840242ac110002','PROJECT_TYPE_PYTORCH','pytorch','PROJECT_TYPE','submarine system dict, Do not modify.',6,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');
INSERT INTO `sys_dict_item` VALUES ('2ed844fae84111e9ab840242ac110002','PROJECT_VISIBILITY_PRIVATE','private','PROJECT_VISIBILITY','submarine system dict, Do not modify.',1,0,NULL,'2019-09-01 01:09:32',NULL,'2019-09-01 01:13:19');
INSERT INTO `sys_dict_item` VALUES ('341d5a35e84111e9ab840242ac110002','PROJECT_VISIBILITY_TEAM','team','PROJECT_VISIBILITY','submarine system dict, Do not modify.',2,0,NULL,'2019-09-01 01:10:33',NULL,'2019-09-01 01:12:53');
INSERT INTO `sys_dict_item` VALUES ('3866b369e84111e9ab840242ac110002','PROJECT_VISIBILITY_PUBLIC','public','PROJECT_VISIBILITY','submarine system dict, Do not modify.',3,0,NULL,'2019-09-01 01:11:29',NULL,'2019-09-01 01:12:47');
INSERT INTO `sys_dict_item` VALUES ('3cc1a373e84111e9ab840242ac110002','PROJECT_PERMISSION_VIEW','can view','PROJECT_PERMISSION','submarine system dict, Do not modify.',1,0,NULL,'2019-09-01 01:09:32',NULL,'2019-09-01 01:13:19');
INSERT INTO `sys_dict_item` VALUES ('44e90f6ce84111e9ab840242ac110002','PROJECT_PERMISSION_EDIT','can edit','PROJECT_PERMISSION','submarine system dict, Do not modify.',2,0,NULL,'2019-09-01 01:11:29',NULL,'2019-09-01 01:12:47');
INSERT INTO `sys_dict_item` VALUES ('40dbb5ece84111e9ab840242ac110002','PROJECT_PERMISSION_EXECUTE','can execute','PROJECT_PERMISSION','submarine system dict, Do not modify.',3,0,NULL,'2019-09-01 01:10:33',NULL,'2019-09-01 01:12:53');

-- ----------------------------
-- Records of sys_department
-- ----------------------------
INSERT INTO `sys_department` VALUES ('e3d69d19c8d211e98edc0242ac110002','A','Company',NULL,0,'','0',NULL,'2019-08-27 13:59:30',NULL,'2019-08-27 14:02:05');
INSERT INTO `sys_department` VALUES ('eec10fe9c8d211e98edc0242ac110002','AA','DepartmentA','A',0,'','0',NULL,'2019-08-27 13:59:48',NULL,'2019-08-27 14:04:11');
INSERT INTO `sys_department` VALUES ('f8b42e19c8d211e98edc0242ac110002','AB','DepartmentB','A',0,'','0',NULL,'2019-08-27 14:00:05',NULL,'2019-08-27 14:07:19');
INSERT INTO `sys_department` VALUES ('13a1916dc8d311e98edc0242ac110002','ABA','GroupA','AB',0,'','0',NULL,'2019-08-27 14:00:50',NULL,'2019-08-27 14:09:21');
INSERT INTO `sys_department` VALUES ('1bc0cd98c8d311e98edc0242ac110002','AAA','GroupB','AA',0,'','0',NULL,'2019-08-27 14:01:04',NULL,'2019-08-29 16:48:56');

-- ----------------------------
-- Records of sys_user
-- ----------------------------
INSERT INTO `sys_user` VALUES ('e9ca23d68d884d4ebb19d07889727dae', 'admin', 'administrator', '21232f297a57a5a743894a0e4a801fc3', 'avatar.png', '2018-12-05 00:00:00', NULL, 'dev@submarine.org', '18566666661', NULL, NULL, NULL, 0, 'admin', '2019-07-05 14:47:22', 'admin', '2019-07-05 14:47:22');

-- ----------------------------
-- Records of team
-- ----------------------------
INSERT INTO `team` VALUES ('e9ca23d68d884d4ebb19d07889721234', 'admin', 'Submarine', 'admin', '2020-05-06 14:00:05', 'Jack', '2020-05-06 14:00:14');

-- ----------------------------
-- Records of environment
-- ----------------------------
INSERT INTO `environment` VALUES
('environment_1600862964725_0001', 'notebook-env', '{"name":"notebook-env","dockerImage":"apache/submarine:jupyter-notebook-0.7.0-SNAPSHOT","kernelSpec":{"name":"submarine_jupyter_py3","channels":["defaults"],"condaDependencies":[],"pipDependencies":[]}}', 'admin', '2020-09-21 14:00:05', 'admin', '2020-09-21 14:00:14'),
('environment_1600862964725_0002', 'notebook-gpu-env', '{"name":"notebook-gpu-env","dockerImage":"apache/submarine:jupyter-notebook-gpu-0.7.0-SNAPSHOT","kernelSpec":{"name":"submarine_jupyter_py3","channels":["defaults"],"condaDependencies":[],"pipDependencies":[]}}', 'admin', '2021-03-28 17:00:00', 'admin', '2021-03-28 17:00:00');

-- ----------------------------
-- Records of experiment_templates
-- ----------------------------
INSERT INTO `experiment_template` (`id`, `experimentTemplate_name`, `experimentTemplate_spec`, `create_by`, `create_time`, `update_by`, `update_time`) VALUES
('experimentTemplate_1599498007985_0013', 'tf-mnist', '{\"name\": \"tf-mnist\", \"author\": \"author\", \"parameters\": [{\"name\": \"learning_rate\", \"value\": \"0.2\", \"required\": \"false\", \"description\": \"The parameter of train mnist.\"}, {\"name\": \"batch_size\", \"value\": \"150\", \"required\": \"false\", \"description\": \"The parameter of train mnist.\"}, {\"name\": \"experiment_name\", \"required\": \"true\", \"description\": \"experiment name, you should change it to avoid duplication with other experiment names.\"}, {\"name\": \"spec.Ps.replicas\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Ps.resourceMap.cpu\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Ps.resourceMap.memory\", \"value\": \"2G\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Worker.replicas\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Worker.resourceMap.cpu\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Worker.resourceMap.memory\", \"value\": \"2G\", \"required\": \"false\", \"description\": \"\"}], \"description\": \"This is a template to run tf-mnist.\", \"experimentSpec\": {\"meta\": {\"cmd\": \"python /var/tf_mnist/mnist_with_summaries.py --log_dir=/train/log --learning_rate={{learning_rate}} --batch_size={{batch_size}}\", \"name\": \"{{experiment_name}}\", \"envVars\": {\"ENV1\": \"ENV1\"}, \"framework\": \"TensorFlow\", \"namespace\": \"default\"}, \"spec\": {\"Ps\": {\"replicas\": 1, \"resources\": \"cpu=1,memory=2G\", \"resourceMap\": {\"cpu\": \"1\", \"memory\": \"1000M\"}}, \"Worker\": {\"replicas\": 1, \"resources\": \"cpu=1,memory=2G\", \"resourceMap\": {\"cpu\": \"1\", \"memory\": \"2G\"}}}, \"environment\": {\"image\": \"apache/submarine:tf-mnist-with-summaries-1.0\"}}}', NULL, '2020-09-10 16:31:32', NULL, '2020-10-19 17:05:21');

INSERT INTO `experiment_template` (`id`, `experimentTemplate_name`, `experimentTemplate_spec`, `create_by`, `create_time`, `update_by`, `update_time`) VALUES('experimentTemplate_1606489231336_0014', 'pytorch-mnist', '{\"name\": \"pytorch-mnist\", \"author\": \"author\", \"parameters\": [{\"name\": \"experiment_name\", \"required\": \"true\", \"description\": \"experiment name\"}, {\"name\": \"spec.Master.replicas\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Master.resourceMap.cpu\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Master.resourceMap.memory\", \"value\": \"1024M\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Worker.replicas\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Worker.resourceMap.cpu\", \"value\": \"1\", \"required\": \"false\", \"description\": \"\"}, {\"name\": \"spec.Worker.resourceMap.memory\", \"value\": \"1024M\", \"required\": \"false\", \"description\": \"\"}], \"description\": \"This is a template to run pytorch-mnist\\n\", \"experimentSpec\": {\"meta\": {\"cmd\": \"python /var/mnist.py --backend gloo\", \"name\": \"{{experiment_name}}\", \"envVars\": {\"ENV_1\": \"ENV1\"}, \"framework\": \"PyTorch\", \"namespace\": \"default\"}, \"spec\": {\"Master\": {\"replicas\": 1, \"resources\": \"cpu=1,memory=1024M\", \"resourceMap\": {\"cpu\": \"1\", \"memory\": \"1024M\"}}, \"Worker\": {\"replicas\": 1, \"resources\": \"cpu=1,memory=1024M\", \"resourceMap\": {\"cpu\": \"1\", \"memory\": \"1024M\"}}}, \"environment\": {\"image\": \"apache/submarine:pytorch-dist-mnist-1.0\"}}}', NULL, '2020-11-29 17:56:10', NULL, '2020-11-29 17:56:10');
