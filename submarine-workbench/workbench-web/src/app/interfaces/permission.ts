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

import { Action } from './action';

export interface ActionEntitySet {
  action: string;
  defaultCheck: boolean;
  describe: string;
}

export class Permission {
  roleId: string;
  permissionId: string;
  permissionName: string;
  dataAccess?: any;
  actionList?: any;
  actions: Action[];
  actionEntitySet: ActionEntitySet[];

  constructor(permission: Permission) {
    this.roleId = permission.roleId;
    this.permissionId = permission.permissionId;
    this.permissionName = permission.permissionName;
    this.dataAccess = permission.dataAccess;
    this.actionList = permission.actionList;
    this.actions = permission.actions;
    this.actionEntitySet = permission.actionEntitySet;
  }
}
