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
DROP TABLE IF EXISTS `sys_dict`;
CREATE TABLE `sys_dict` (
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
DROP TABLE IF EXISTS `sys_dict_item`;
CREATE TABLE `sys_dict_item` (
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
DROP TABLE IF EXISTS `sys_department`;
CREATE TABLE `sys_department` (
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
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user` (
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
DROP TABLE IF EXISTS `sys_message`;
CREATE TABLE `sys_message` (
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
DROP TABLE IF EXISTS `team`;
CREATE TABLE `team` (
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
DROP TABLE IF EXISTS `team_member`;
CREATE TABLE `team_member` (
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
DROP TABLE IF EXISTS `project`;
CREATE TABLE `project` (
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
DROP TABLE IF EXISTS `project_files`;
CREATE TABLE `project_files` (
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
