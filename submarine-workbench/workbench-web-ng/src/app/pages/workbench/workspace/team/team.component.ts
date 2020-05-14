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
import { FormGroup, FormControl, Validators, ValidationErrors } from '@angular/forms';
import { SysTeam } from '@submarine/interfaces';
import { TeamService } from '@submarine/services';
import { NzMessageService } from 'ng-zorro-antd';



@Component({
  selector: 'app-team',
  templateUrl: './team.component.html',
  styleUrls: ['./team.component.scss']
})
export class TeamComponent implements OnInit {
  column: string = 'createdTime';
  order: string = 'description';
  teamName: string = "";
  teamList: SysTeam[] = [];

  //Form
  newTeamForm: FormGroup; //For Adding Form
  formTeamNameErrMesg = "";

  //Drawer
  drawerVisible = false;
  submitBtnIsLoading = false;

  //Modal
  overviewModalVisible = false;
  currentTeam_teamName: string;
  currentTeam_id: string;
  currentTeam_owner: string;
  currentTeam_createTime: string;

  constructor( 
    private teamService: TeamService,
    private nzMessageService: NzMessageService,
    ) { }

  ngOnInit() {
    this.newTeamForm = new FormGroup({
      'teamName': new FormControl(null, [Validators.required, this.teamNameRequired.bind(this)], this.teamNameCheck.bind(this)),
      'owner': new FormControl(null, Validators.required)
    });

    this.getTeamList();
    
  }
  
  getTeamList() {
    this.teamService.getTeamList({
      column: this.column,
      order: this.order,
      teamName: this.teamName,
    }).subscribe(({ records }) => {
      this.teamList = records;
      console.log(records);
    })
  }

  submitNewTeam() {
    this.submitBtnIsLoading = true;
    this.teamService.createTeam({
      teamName: this.newTeamForm.get('teamName').value,
      owner: this.newTeamForm.get('owner').value,
      createBy: this.newTeamForm.get('owner').value
    }).subscribe(
      () => {
      this.nzMessageService.success('Create team success!');
      this.getTeamList();
      this.drawerVisible = false ;
      this.submitBtnIsLoading = false ;
      this.newTeamForm.reset();
    }, err => {
      this.nzMessageService.error(err.message);
      this.submitBtnIsLoading = false ;
    })
  }

  deleteTeam(teamData: SysTeam) {
    this.teamService.deleteTeam(teamData.id).subscribe(
      () => {
        this.nzMessageService.success('Delete team success!');
        this.getTeamList();
      }
      , err => {
        this.nzMessageService.error(err.message);
      }
    )
  }

  closeDrawer() {
    this.drawerVisible = false ;
    this.newTeamForm.reset({
    });
  }

  addTeam(){
    this.drawerVisible = true ;
  }

  teamNameRequired(check: FormControl): {[key: string]:any}|null{
    if (check.value === "") {
      var errorMessage = "Please enter new team name!";
      this.formTeamNameErrMesg = errorMessage;
      return {mesg: true}
    }
    else {
      this.formTeamNameErrMesg = "";
      return null;
    }
  }

  teamNameCheck(check: FormControl): Promise<ValidationErrors|null>{
    var params = {
      tableName: 'team',
      fieldName: 'team_name',
      fieldVal: check.value
    }
    const promise = new Promise((resolve, reject) => {
      this.teamService.newTeamNameCheck(params).then((success) => {
        if (success) {
          resolve(null);
        }
        else {
          this.formTeamNameErrMesg = "This value already exists is not available!";
          resolve({"Duplicate Name": true});
        }
      },(err)=>{
        reject(err);
      });
    });
    return promise;
  }

  showOverview(team) {
    this.overviewModalVisible = true ;
    this.currentTeam_teamName = team.teamName;
    this.currentTeam_id = team.id;
    this.currentTeam_owner = team.owner;
    this.currentTeam_createTime = team.createTime;
  }

  handleOk() {
    this.overviewModalVisible = false ;
  }

  handleCancel() {
    this.overviewModalVisible = false ;
  }
}
