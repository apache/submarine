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
-- Table structure for sys_user
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user` (
  `id` varchar(32) NOT NULL COMMENT '主键id',
  `name` varchar(100) default NULL COMMENT '真实姓名',
  `username` varchar(100) default NULL COMMENT '登录账号',
  `password` varchar(255) default NULL COMMENT '密码',
  `avatar` varchar(255) default NULL COMMENT '头像',
  `birthday` datetime default NULL COMMENT '生日',
  `sex` int(1) default NULL COMMENT '性别(1：男, 2：女)',
  `email` varchar(32) default NULL COMMENT '电子邮件',
  `phone` varchar(32) default NULL COMMENT '电话',
  `org_code` varchar(64) default NULL COMMENT '部门code',
  `status` int(1) default NULL COMMENT '状态(1：正常  0：冻结)',
  `deleted` int(1) default NULL COMMENT '删除状态(1，正常，0已删除)',
  `lastLoginIp` varchar(32) default NULL COMMENT 'last login ip',
  `lastLoginTime` datetime default NULL COMMENT 'last login time',
  `create_by` varchar(32) default NULL COMMENT '创建人',
  `create_time` datetime default NULL COMMENT '创建时间',
  `update_by` varchar(32) default NULL COMMENT '更新人',
  `update_time` datetime default NULL COMMENT '更新时间',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `sys_user_name` USING BTREE (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='system user';

-- ----------------------------
-- Table structure for sys_dict
-- ----------------------------
DROP TABLE IF EXISTS `sys_dict`;
CREATE TABLE `sys_dict` (
  `id` varchar(32) NOT NULL,
  `dict_code` varchar(100) default NULL COMMENT '字典编码',
  `dict_name` varchar(100) default NULL COMMENT '字典名称',
  `description` varchar(255) default NULL COMMENT '描述',
  `deleted` int(1) default 0 COMMENT '删除状态(0正常，1已删除)',
  `type` int(1) default 0 COMMENT '字典类型(0为string,1为number)',
  `create_by` varchar(32) default NULL COMMENT '创建人',
  `create_time` datetime default NULL COMMENT '创建时间',
  `update_by` varchar(32) default NULL COMMENT '更新人',
  `update_time` datetime default NULL COMMENT '更新时间',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `sys_dict_dict_code` USING BTREE (`dict_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Table structure for sys_dict_item
-- ----------------------------
DROP TABLE IF EXISTS `sys_dict_item`;
CREATE TABLE `sys_dict_item` (
  `id` varchar(32) NOT NULL,
  `dict_id` varchar(32) default NULL COMMENT '字典id',
  `item_text` varchar(100) default NULL COMMENT '字典项文本',
  `item_value` varchar(100) default NULL COMMENT '字典项值',
  `description` varchar(255) default NULL COMMENT '描述',
  `sort_order` int(3) default 0 COMMENT '排序',
  `deleted` int(1) default 0 COMMENT '删除状态(0正常，1已删除)',
  `create_by` varchar(32) default NULL,
  `create_time` datetime default NULL,
  `update_by` varchar(32) default NULL,
  `update_time` datetime default NULL,
  PRIMARY KEY  (`id`),
  CONSTRAINT `FK_SYS_DICT_ITEM_DICT_ID` FOREIGN KEY (`dict_id`) REFERENCES `sys_dict` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
