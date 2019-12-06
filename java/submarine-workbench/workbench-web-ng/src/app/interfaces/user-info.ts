/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { Role } from './role';

export class UserInfo {
  id: string;
  name: string;
  username: string;
  password: string;
  avatar: string;
  status: number;
  telephone: string;
  lastLoginIp: string;
  lastLoginTime: number;
  creatorId: string;
  createTime: number;
  merchantCode: string;
  deleted: number;
  roleId: string;
  role: Role;

  constructor(res: UserInfo) {
    this.id = res.id;
    this.name = res.name;
    this.username = res.username;
    this.password = res.password;
    this.avatar = res.avatar;
    this.status = res.status;
    this.telephone = res.telephone;
    this.lastLoginIp = res.lastLoginIp;
    this.lastLoginTime = res.lastLoginTime;
    this.creatorId = res.creatorId;
    this.createTime = res.createTime;
    this.merchantCode = res.merchantCode;
    this.deleted = res.deleted;
    this.roleId = res.roleId;
    this.role = new Role(res.role);
  }
}
