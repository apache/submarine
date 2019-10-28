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

import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { SysUser } from '@submarine/interfaces';
import { SysDeptSelect } from '@submarine/interfaces/sys-dept-select';
import { DepartmentService, UserService } from '@submarine/services';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-manager-user',
  templateUrl: './user.component.html',
  styleUrls: ['./user.component.scss']
})
export class UserComponent implements OnInit {
  column: string = 'createdTime';
  order: string = 'desc';
  field = [
    'id',
    'userName',
    'realName',
    'deptName',
    'roleCode',
    'status@dict',
    'sex@dict',
    'email',
    'createTime',
    'action'
  ];
  accountName: string = '';
  email: string = '';
  deptCode: string = '';
  pageNo: number = 1;
  pageSize: number = 10;
  userList: SysUser[] = [];
  total: number = 0;
  sysDeptTreeList: SysDeptSelect[] = [];
  form: FormGroup;
  resetPasswordModalVisible: boolean = false;
  currentSysUser: SysUser;
  userDrawerVisible: boolean = false;
  private userDrawerReadonly: boolean = false;

  constructor(
    private userService: UserService,
    private deptService: DepartmentService,
    private fb: FormBuilder,
    private nzMessageService: NzMessageService
  ) {
  }

  ngOnInit() {
    this.fetchUserList();

    this.deptService.fetchSysDeptSelect().subscribe(list => {
      this.sysDeptTreeList = list;
    });

    this.form = this.fb.group({
      deptCode: [this.deptCode],
      accountName: [this.accountName],
      email: [this.email]
    });
  }

  queryUserList() {
    const { deptCode, accountName, email } = this.form.getRawValue();
    this.deptCode = deptCode;
    this.accountName = accountName;
    this.email = email;

    this.fetchUserList();
  }

  fetchUserList() {
    this.userService
      .fetchUserList({
        column: this.column,
        order: this.order,
        field: this.field.join(','),
        accountName: this.accountName,
        email: this.email,
        deptCode: this.deptCode ? this.deptCode : '',
        pageNo: '' + this.pageNo,
        pageSize: '' + this.pageSize
      })
      .subscribe(({ records, total }) => {
        this.total = total;
        this.userList = records;
      });
  }

  onShowResetPasswordModal(data: SysUser) {
    this.currentSysUser = data;
    this.resetPasswordModalVisible = true;
  }

  onHideResetPasswordModal() {
    this.currentSysUser = null;
    this.resetPasswordModalVisible = false;
  }

  onChangePassword(password: string) {
    const { id } = this.currentSysUser;

    this.resetPasswordModalVisible = false;

    this.userService.changePassword(id, password).subscribe(() => {
      this.nzMessageService.success('Change password success!');
    }, () => {
      this.nzMessageService.error('Change password error');
    });
  }

  onShowUserDrawer(sysUser?: SysUser, readOnly = false) {
    this.currentSysUser = sysUser;
    this.userDrawerReadonly = readOnly;
    this.userDrawerVisible = true;
  }

  onCloseUserDrawer() {
    this.userDrawerVisible = false;
  }

  onSubmitUserDrawer(formData: Partial<SysUser>) {
    this.userDrawerVisible = false;

    if (formData.id) {
      this.userService.updateUser(formData).subscribe(() => {
        this.nzMessageService.success('Update user success!');
        this.queryUserList();
      }, err => {
        this.nzMessageService.error(err.message);
      });
    } else {
      this.userService.createUser(formData).subscribe(() => {
        this.nzMessageService.success('Add user success!');
        this.queryUserList();
      }, err => {
        this.nzMessageService.error(err.message);
      });
    }
  }

  onDeleteUser(data: SysUser) {
    this.userService.deleteUser(data.id).subscribe(
      () => {
        this.nzMessageService.success('Delete user success!');
        this.fetchUserList();
      }, err => {
        this.nzMessageService.success(err.message);
      }
    );
  }
}
