<!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->

# Database Design

Submarine needs to use the database to store information about the organization, user, projects, tasks, and configuration of the system information, So consider using mysql to store this data.

+ MySQL will be included in the `mini-submarine` docker image to allow users to quickly experience the `submarine workbench`.
+ In a production environment, the `submarine workbench` can be connected to the official mysql database.

## Prerequisite

Must:

- MySQL
- MyBatis



## Docker

developmenet and test

1. Run docker

```
docker run -p 3306:3306 -d --name mysql -e MYSQL_ROOT_PASSWORD=password mysql:5.7.27
docker exec -it mysql bash
```

2. Create mysql user submarine

```
# in mysql container
mysql -uroot -ppassword
CREATE USER 'submarine'@'%' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON * . * TO 'submarine'@'%';
quit
```

3. Create submarine database

```
root:> mysql -u submarine -p -Dsubmarine
mysql> create database submarine;
```



## Table Schema

### sys_dict

```
DROP TABLE IF EXISTS `sys_dict`;
CREATE TABLE `sys_dict` (
  `id` varchar(32) NOT NULL,
  `dict_name` varchar(100) default NULL COMMENT '字典名称',
  `dict_code` varchar(100) default NULL COMMENT '字典编码',
  `description` varchar(255) default NULL COMMENT '描述',
  `del_flag` int(1) default NULL COMMENT '删除状态',
  `create_by` varchar(32) default NULL COMMENT '创建人',
  `create_time` datetime default NULL COMMENT '创建时间',
  `update_by` varchar(32) default NULL COMMENT '更新人',
  `update_time` datetime default NULL COMMENT '更新时间',
  `type` int(1) unsigned zerofill default '0' COMMENT '字典类型0为string,1为number',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `indextable_dict_code` USING BTREE (`dict_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

### sys_dict_item

```
DROP TABLE IF EXISTS `sys_dict_item`;
CREATE TABLE `sys_dict_item` (
  `id` varchar(32) NOT NULL,
  `dict_id` varchar(32) default NULL COMMENT '字典id',
  `item_text` varchar(100) default NULL COMMENT '字典项文本',
  `item_value` varchar(100) default NULL COMMENT '字典项值',
  `description` varchar(255) default NULL COMMENT '描述',
  `sort_order` int(10) default NULL COMMENT '排序',
  `status` int(1) default NULL COMMENT '状态（1启用 0不启用）',
  `create_by` varchar(32) default NULL,
  `create_time` datetime default NULL,
  `update_by` varchar(32) default NULL,
  `update_time` datetime default NULL,
  PRIMARY KEY  (`id`),
  KEY `index_table_dict_id` USING BTREE (`dict_id`),
  KEY `index_table_sort_order` USING BTREE (`sort_order`),
  KEY `index_table_dict_status` USING BTREE (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

### sys_depart

```
DROP TABLE IF EXISTS `sys_depart`;
CREATE TABLE `sys_depart` (
  `id` varchar(32) NOT NULL COMMENT 'ID',
  `parent_id` varchar(32) default NULL COMMENT '父机构ID',
  `depart_name` varchar(100) NOT NULL COMMENT '机构/部门名称',
  `depart_name_en` varchar(500) default NULL COMMENT '英文名',
  `depart_name_abbr` varchar(500) default NULL COMMENT '缩写',
  `depart_order` int(11) default '0' COMMENT '排序',
  `description` text COMMENT '描述',
  `org_type` varchar(10) default NULL COMMENT '机构类型 1一级部门 2子部门',
  `org_code` varchar(64) NOT NULL COMMENT '机构编码',
  `mobile` varchar(32) default NULL COMMENT '手机号',
  `fax` varchar(32) default NULL COMMENT '传真',
  `address` varchar(100) default NULL COMMENT '地址',
  `memo` varchar(500) default NULL COMMENT '备注',
  `status` varchar(1) default NULL COMMENT '状态（1启用，0不启用）',
  `del_flag` varchar(1) default NULL COMMENT '删除状态（0，正常，1已删除）',
  `create_by` varchar(32) default NULL COMMENT '创建人',
  `create_time` datetime default NULL COMMENT '创建日期',
  `update_by` varchar(32) default NULL COMMENT '更新人',
  `update_time` datetime default NULL COMMENT '更新日期',
  PRIMARY KEY  (`id`),
  KEY `index_depart_parent_id` USING BTREE (`parent_id`),
  KEY `index_depart_depart_order` USING BTREE (`depart_order`),
  KEY `index_depart_org_code` USING BTREE (`org_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='组织机构表';
```

### sys_user

```
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
  `del_flag` int(1) default NULL COMMENT '删除状态（0，正常，1已删除）',
  `lastLoginIp` varchar(32) default NULL COMMENT 'last login ip',
  `lastLoginTime` datetime default NULL COMMENT 'last login time',
  `create_by` varchar(32) default NULL COMMENT '创建人',
  `create_time` datetime default NULL COMMENT '创建时间',
  `update_by` varchar(32) default NULL COMMENT '更新人',
  `update_time` datetime default NULL COMMENT '更新时间',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `index_user_name` USING BTREE (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='system user';
```
