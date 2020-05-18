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

import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { UserService, ProjectService  } from '@submarine/services';
import { UserInfo } from '@submarine/interfaces';
import { NzNotificationService } from 'ng-zorro-antd';
import { tap } from 'rxjs/operators';
import { Observable } from 'rxjs';

@Component({
  selector: 'submarine-project',
  templateUrl: './project.component.html',
  styleUrls: ['./project.component.scss']
})
export class ProjectComponent implements OnInit {
  newProject = false;
  existProjects = [];
  isLoading = false;
  username = '';
  @ViewChild('inputElement', { static: false }) inputElement: ElementRef;
  userInfo$: Observable<UserInfo>;

  constructor(
    private projectService: ProjectService,
    private userService: UserService,
    private nzNotificationService: NzNotificationService
  ) {

  }

  //TODO(jasoonn): get projects data from server
  async ngOnInit() {
    await this.userService.fetchUserInfo().toPromise().then(data => {
      this.username = data.username;

    });
    //TODO(chiajoukuo): add pagination
    var params = {
      userName: this.username,
      column: 'update_time',
      order: 'desc',
      pageNo: ''+1,//this.pagination.current,
      pageSize: ''+99//this.pagination.pageSize
    }
    var res;
    this.projectService.fetchProjectList(params)
    .subscribe(
      (data) => {
        res = data['records']
        for(var i=0; i<res.length; i++){
          this.existProjects.push({
            projectName: res[i].name,
            description: res[i].description,
            tags: res[i].tags ===null?[]:res[i].tags, //['12', 'Tag 2']
            inputTagVisibility: false,
            projectInputTag: '',
            starNum: res[i].starNum,
            likeNum: res[i].likeNum,
            msgNum: res[i].messageNum
          })
        }
      },
      error => {
        console.log("ERROR", error)
      }
    );

  }

  addProject(event){
    this.existProjects.push({
      projectName: event.name,
      description: event.description,
      tags: [],
      inputTagVisibility: false,
      projectInputTag: '',
      starNum: 0,
      likeNum: 0,
      msgNum: 0
    })

    this.projectService.addProject(event).subscribe(() => {
    }, err => {
      console.log("ERROR", err)
    });
    this.newProject = false;
    console.log("proj", event);
  }


  //TODO(jasoonn): Update tag in server
  handleCloseTag(project, tag){
    project.tags = project.tags.filter(itag => itag!==tag);
    console.log(project);
    console.log(tag);
  }
  // TODO(jasoonn): update tag in server
  handleInputConfirm(project): void {
    console.log(project.tags);
    if (project.projectInputTag && (project.tags == null || (project.tags != null && project.tags.indexOf(project.projectInputTag)=== -1))) {
      console.log(project);
      project.tags = [...project.tags, project.projectInputTag];
    }
    project.inputTagVisibility = false;
    project.projectInputTag = '';
  }

  showInput(project): void {
    project.inputTagVisibility = true;
    setTimeout(() => {
      this.inputElement.nativeElement.focus();
    }, 10);
  }
}
