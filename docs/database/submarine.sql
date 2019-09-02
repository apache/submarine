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
  `dict_name` varchar(100) NOT NULL COMMENT 'dict name',
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
  `item_name` varchar(100) NOT NULL COMMENT 'dict item name',
  `dict_code` varchar(32) NOT NULL COMMENT 'dict code',
  `description` varchar(255) default NULL COMMENT 'description',
  `sort_order` int(3) default 0 COMMENT 'sort order',
  `deleted` int(1) default 0 COMMENT 'delete status(0:normal,1:already deleted)',
  `create_by` varchar(32) default NULL,
  `create_time` datetime default NULL,
  `update_by` varchar(32) default NULL,
  `update_time` datetime default NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `UK_SYS_DICT_ITEM_CODE` (`item_code`),
  CONSTRAINT `FK_SYS_DICT_ITEM_DICT_CODE` FOREIGN KEY (`dict_code`) REFERENCES `sys_dict` (`dict_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for system department
-- ----------------------------
DROP TABLE IF EXISTS `sys_department`;
CREATE TABLE `sys_department` (
  `id` varchar(32) NOT NULL COMMENT 'ID',
  `dept_code` varchar(32) NOT NULL COMMENT 'department code',
  `dept_name` varchar(100) NOT NULL COMMENT 'department name',
  `parent_code` varchar(32) default NULL COMMENT 'parent dept code',
  `sort_order` int(3) default 0 COMMENT 'sort order',
  `description` text COMMENT 'description',
  `deleted` varchar(1) default 0 COMMENT 'delete status(0:normal,1:already deleted)',
  `create_by` varchar(32) default NULL COMMENT 'create user',
  `create_time` datetime default NULL COMMENT 'create time',
  `update_by` varchar(32) default NULL COMMENT 'last update user',
  `update_time` datetime default NULL COMMENT 'last update time',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `UK_DEPT_CODE` (`dept_code`),
  CONSTRAINT `FK_SYS_DEPT_PARENT_CODE` FOREIGN KEY (`parent_code`) REFERENCES `sys_department` (`dept_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for sys_user
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user` (
  `id` varchar(32) NOT NULL COMMENT 'id',
  `user_name` varchar(100) NOT NULL COMMENT 'login name',
  `real_name` varchar(100) NOT NULL COMMENT 'real name',
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
  UNIQUE KEY `sys_user_name` (`user_name`),
  CONSTRAINT `FK_SYS_USER_DEPT_CODE` FOREIGN KEY (`dept_code`) REFERENCES `sys_department` (`dept_code`),
  CONSTRAINT `FK_SYS_USER_SEX` FOREIGN KEY (`sex`) REFERENCES `sys_dict_item` (`item_code`),
  CONSTRAINT `FK_SYS_USER_STATUS` FOREIGN KEY (`status`) REFERENCES `sys_dict_item` (`item_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
