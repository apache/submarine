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
  `sex` int(1) default NULL COMMENT '性别（1：男, 2：女）',
  `email` varchar(32) default NULL COMMENT '电子邮件',
  `phone` varchar(32) default NULL COMMENT '电话',
  `org_code` varchar(64) default NULL COMMENT '部门code',
  `status` int(1) default NULL COMMENT '状态(1：正常  2：冻结 ）',
  `deleted` int(1) default NULL COMMENT '删除状态（1，正常，2已删除）',
  `lastLoginIp` varchar(32) default NULL COMMENT 'last login ip',
  `lastLoginTime` datetime default NULL COMMENT 'last login time',
  `create_by` varchar(32) default NULL COMMENT '创建人',
  `create_time` datetime default NULL COMMENT '创建时间',
  `update_by` varchar(32) default NULL COMMENT '更新人',
  `update_time` datetime default NULL COMMENT '更新时间',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `index_user_name` USING BTREE (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='system user';
