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
import { NzMessageService } from 'ng-zorro-antd/message';
import { NzNotificationService } from 'ng-zorro-antd/notification';

@Component({
  selector: 'app-team',
  templateUrl: './team.component.html',
  styleUrls: ['./team.component.scss']
})
export class TeamComponent implements OnInit {  

  // Get into a Team or not
  isEntering = false;

  //Get Current Team Data (Now Use Simulated Data) 
  currentTeamName: string;
  currentTeamDepartment: string;
  currentTeamDescription: string;
  currentTeamProjectNum: Number;
  currentTeamMemberNum: Number;
  currentTeamSettingPermission: boolean;

  // Is Editging Members or not
  isEditingMember = false;

  // For AddUserModal
  isVisible = false;
  isOkLoading = false;

  // Simulated Data for Members
  member = [
    {
      name: 'Anna',
      email: 'test@mail.com',
      permission: 'admin'
    },
    {
      name: 'James',
      email: 'test@mail.com',
      permission: 'admin'
    },
    {
      name: 'Jack',
      email: 'test@mail.com',
      permission: 'contributer'
    },
    {
      name: 'Ken',
      email: 'test@mail.com',
      permission: 'viwer'
    }
  ]

  // Simulated Data for Teams
  existTeams = [
    {
      name: "Submarine",
      department: "Apache",
      description: "Something about this team...",
      projectNum: 3,
      memberNum: 3,
      role: 'admin',
      settingPermission: true
    },
    {
      name: "Team2",
      department: "Apple",
      description: "Something about this team...",
      projectNum: 3,
      memberNum: 3,
      role: 'viwer',
      settingPermission: false
    },
  ]
  constructor(
    private nzMessageService: NzMessageService, 
    private notification: NzNotificationService,
    ) { }

  ngOnInit() {
  }

  enter(team) {
    this.isEntering = true;
    this.currentTeamName = team.name;
    this.currentTeamDepartment = team.department;
    this.currentTeamDescription = team.description;
    this.currentTeamProjectNum = team.projectNum;
    this.currentTeamMemberNum = team.memberNum;
    this.currentTeamSettingPermission = team.settingPermission;
  }

  startEditMember() {
    this.isEditingMember=true;
  }

  saveEditMember() {
    this.isEditingMember=false;
  }

  cancel(): void {
  }

  confirm(): void {
    this.nzMessageService.info('Delete Successful!');
  }

  // For AddUserModal
  startAddUser(): void {
    this.isVisible = true;
  }

  // Add Success
  createNotification(type: string): void {
    this.notification.create(
      type,
      'Add Successful!',
      'Make sure thar user check invitation!'
    );
  }
  
  // For AddUserModal
  handleOk(): void {
    this.isOkLoading = true;
    setTimeout(() => {
      this.isVisible = false;
      this.isOkLoading = false;
      this.createNotification("success");
    }, 1000);
  }

  // For AddUserModal
  handleCancel(): void {
    this.isVisible = false;
  }

  //TODO:kobe860219 ; Get Team From DataBase
  getTeamDataFromDB() {

  }

  //TODO:kobe860219 ; Get User From DataBase
  getUserDataFromDB() {
  }

  //TODO:kobe860219 ; Update Data to DataBase
  updateTeamData() {

  }
}