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
import { FormBuilder, FormGroup, FormControl } from '@angular/forms';
import { SysTeam } from '@submarine/interfaces';
import { TeamService } from '@submarine/services';

import { Observable } from 'rxjs';



@Component({
  selector: 'app-team',
  templateUrl: './team.component.html',
  styleUrls: ['./team.component.scss']
})
export class TeamComponent implements OnInit {  

  //
  column: string = 'createdTime';
  order: string = 'description';
  teamName: string = '';
  pageNo: number = 1;
  pageSize: number = 10;
  teamList: SysTeam[] = [];

  //Form
  searchTeamForm: FormGroup; //For Searching
  newTeamForm: FormGroup; //For Adding Form

  drawerVisible = false;

  constructor( 
    private teamService: TeamService,
    ) { }

  ngOnInit() {
    this.searchTeamForm = new FormGroup({'teamName': new FormControl});

    this.getTeamList();
    
  }
  
  getTeamList() {
    this.teamService.getTeamList({
      column: this.column,
      order: this.order,
      teamName: this.teamName,
      pageNo: '' + this.pageNo,
      pageSize: '' + this.pageSize
    }).subscribe(({ records }) => {
      this.teamList = records;
      console.log(records);
    })
  }

  closeDrawer() {
    this.drawerVisible = false ;
  }
  addTeam(){
    this.drawerVisible = true ;
  }
  queryTeam(){}
}
