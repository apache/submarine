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

import { Component, OnInit, ViewChild, Output, EventEmitter, Input } from '@angular/core';
import { NgForm } from '@angular/forms';


@Component({
  selector: 'app-new-project-page',
  templateUrl: './new-project-page.component.html',
  styleUrls: ['./new-project-page.component.scss']
})
export class NewProjectPageComponent implements OnInit {
  @Output() closeProjectPage = new EventEmitter<boolean>();
  @ViewChild('f', { static: true }) signupForm: NgForm;
  //Todo: get team from API
  teams = ['ciil'];
  
  current = 0;
  
  newProjectContent = { projectName: '', description: '', visibility: 'Private', team: '' ,permission: 'View', dataSet: []};
  

  constructor() { }

  ngOnInit() {
  }


  clearProject(){
    this.closeProjectPage.emit(true);
  }

  pre(): void {
    this.current -= 1;
  }

  next(): void {
    this.current += 1;
  }

  //Todo : Add the new project
  done(): void{
    console.log(this.newProjectContent);
    this.clearProject();
  }

  //Todo : open in notebook
  openNotebook() {
    ;
  }
}
