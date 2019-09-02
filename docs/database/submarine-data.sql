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
-- Records of sys_dict
-- ----------------------------
INSERT INTO `sys_dict` VALUES ('ca2dd544ca4c11e9a71e0242ac110002','SYS_USER_SEX','Sys user sex','submarine system dict, Do not modify.',0,0,NULL,'2019-08-29 11:04:36',NULL,'2019-09-01 01:08:12');
INSERT INTO `sys_dict` VALUES ('f405a7b1cc5411e9af810242ac110002','SYS_USER_STATUS','Sys user status','submarine system dict, Do not modify.',0,0,NULL,'2019-09-01 01:08:05',NULL,'2019-09-01 01:08:05');

-- ----------------------------
-- Records of sys_dict_item
-- ----------------------------
INSERT INTO `sys_dict_item` VALUES ('27ef1080cc5511e9af810242ac110002','SYS_USER_STATUS_AVAILABLE','Available','SYS_USER_STATUS','submarine system dict, Do not modify.',1,0,NULL,'2019-09-01 01:09:32',NULL,'2019-09-01 01:13:19');
INSERT INTO `sys_dict_item` VALUES ('4c5d736acc5511e9af810242ac110002','SYS_USER_STATUS_LOCKED','Locked','SYS_USER_STATUS','submarine system dict, Do not modify.',2,0,NULL,'2019-09-01 01:10:33',NULL,'2019-09-01 01:12:53');
INSERT INTO `sys_dict_item` VALUES ('6d5ae3b2cc5511e9af810242ac110002','SYS_USER_STATUS_REGISTERED','New Registered','SYS_USER_STATUS','submarine system dict, Do not modify.',3,0,NULL,'2019-09-01 01:11:29',NULL,'2019-09-01 01:12:47');
INSERT INTO `sys_dict_item` VALUES ('d018e2b0ca4c11e9a71e0242ac110002','SYS_USER_SEX_MALE','Male','SYS_USER_SEX','submarine system dict, Do not modify.',1,0,NULL,'2019-08-29 11:04:46',NULL,'2019-09-01 00:53:54');
INSERT INTO `sys_dict_item` VALUES ('d94410adca4c11e9a71e0242ac110002','SYS_USER_SEX_FEMALE','Female','SYS_USER_SEX','submarine system dict, Do not modify.',2,0,NULL,'2019-08-29 11:05:02',NULL,'2019-09-01 00:54:00');

-- ----------------------------
-- Records of sys_department
-- ----------------------------
INSERT INTO `sys_department` VALUES ('e3d69d19c8d211e98edc0242ac110002','A','Company',NULL,0,'','0',NULL,'2019-08-27 13:59:30',NULL,'2019-08-27 14:02:05');
INSERT INTO `sys_department` VALUES ('eec10fe9c8d211e98edc0242ac110002','AA','DeptartmentA','A',0,'','0',NULL,'2019-08-27 13:59:48',NULL,'2019-08-27 14:04:11');
INSERT INTO `sys_department` VALUES ('f8b42e19c8d211e98edc0242ac110002','AB','DepartmentB','A',0,'','0',NULL,'2019-08-27 14:00:05',NULL,'2019-08-27 14:07:19');
INSERT INTO `sys_department` VALUES ('13a1916dc8d311e98edc0242ac110002','ABA','GroupA','AB',0,'','0',NULL,'2019-08-27 14:00:50',NULL,'2019-08-27 14:09:21');
INSERT INTO `sys_department` VALUES ('1bc0cd98c8d311e98edc0242ac110002','AAA','GroupB','AA',0,'','0',NULL,'2019-08-27 14:01:04',NULL,'2019-08-29 16:48:56');

-- ----------------------------
-- Records of sys_user
-- ----------------------------
INSERT INTO `sys_user` VALUES ('e9ca23d68d884d4ebb19d07889727dae', 'admin', 'administrator', '21232f297a57a5a743894a0e4a801fc3', 'avatar.png', '2018-12-05 00:00:00', NULL, 'dev@submarine.org', '18566666661', NULL, NULL, NULL, 1, 'admin', '2019-07-05 14:47:22', 'admin', '2019-07-05 14:47:22');
