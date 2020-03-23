/*!
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
import { FormGroup, FormControl, Validators } from '@angular/forms';

@Component({
  selector: 'submarine-data',
  templateUrl: './data.component.html',
  styleUrls: ['./data.component.scss']
})
export class DataComponent implements OnInit {

  listOfDatabaseValue = ['default'];
  listofDatabase = ['db1', 'db2', 'default'];
  searchValue = "";

  listData= [
    {
      name: 'col-0',
      type: 'string',
      comment: 'comment...',
      nameTmp: 'col-0',
      typeTmp: 'string',
      commentTmp: 'comment...',
      edit: false
    },
    {
      name: 'col-1',
      type: 'string',
      comment: 'comment...',
      nameTmp: 'col-1',
      typeTmp: 'string',
      commentTmp: 'comment...',
      edit: false
    }
  ];
  listCount = 2;

  //Create table part
  newTable = false;
  newTablePage = 0;
  createTable: FormGroup;
  listCreateData = [];

  constructor() { }

  ngOnInit() {
    this.createTable =  new FormGroup({
      'dataSource': new FormControl("upload"),
      'path': new FormControl(null),
      'uploadFile': new FormControl(null),
      'fileType': new FormControl(null),
      'columnDelimeter': new FormControl('.', [Validators.required]),
      'header': new FormControl('false'),
      
      'dataBaseName': new FormControl('db1'),
      'tableName': new FormControl(null, [Validators.required])
    });
  }

  //TODO(jasoonn): Perform sorting
  listSort(){
    console.log('sort list according to ' + this.searchValue);
  }

  cancelEdit(data){
    data.nameTmp = data.name;
    data.typeTmp = data.type;
    data.commentTmp = data.comment;
    data.edit = false;
  }

  //TODO(jasoonn): Update remote database
  saveEdit(data){
    data.name = data.nameTmp;
    data.type = data.typeTmp;
    data.comment = data.commentTmp;
    data.edit = false;
  }

  addCol(){
    this.listData.push(
      {
        name: 'col-' + this.listCount,
        type: 'string',
        comment: 'comment...',
        nameTmp: 'col-' + this.listCount,
        typeTmp: 'string',
        commentTmp: 'comment...',
        edit: false
      }
    );
    this.listData=[...this.listData];
    this.listCount ++;
  }

  removeCol(name){
    this.listData = this.listData.filter(d => d.name !== name);
  }

  //TODO(jasoonn): Create Table in Notebook
  openNotebook(){
    this.newTable = false;
    this.newTablePage = 0;
  }

  //TODO(jasoonn): Parse column while creating Table
  parseColumn(){
    ;
  }

  addCreateCol(){
    this.listCreateData.push(
      {
        name: 'col_' + this.listCreateData.length,
        type: 'string',
        comment: 'comment...',
        nameTmp: 'col_' + this.listCreateData.length,
        typeTmp: 'string',
        commentTmp: 'comment...',
        edit: false
      }
    );
    this.listCreateData=[...this.listCreateData];
  }

  removeCreateCol(name){
    this.listCreateData = this.listCreateData.filter(d => d.name !== name);
  }

  cancelCreateEdit(data){
    data.nameTmp = data.name;
    data.typeTmp = data.type;
    data.commentTmp = data.comment;
    data.edit = false;
  }

  //TODO(jasoonn): Update remote database
  saveCreateEdit(data){
    data.name = data.nameTmp;
    data.type = data.typeTmp;
    data.comment = data.commentTmp;
    data.edit = false;
  }

  //TODO(jasoonn): Create table
  submit(){
    this.newTable = false;
    this.newTablePage = 0;
    console.log(this.createTable);
    this.createTable =  new FormGroup({
      'dataSource': new FormControl("upload"),
      'path': new FormControl(null),
      'uploadFile': new FormControl(null),
      'fileType': new FormControl(null),
      'columnDelimeter': new FormControl('.', [Validators.required]),
      'header': new FormControl('false'),
      
      'dataBaseName': new FormControl('db1'),
      'tableName': new FormControl(null, [Validators.required])
    });
    this.listCreateData = [];
  }
}
