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
import { FormBuilder, FormControl, FormGroup } from '@angular/forms';
import { CascaderOption } from 'ng-zorro-antd/cascader';

interface InterpreterInfo {
  name: string;
  type: string;
  status: string;
  progress: number;
  lastUpdated: number;
}

@Component({
  selector: 'submarine-interpreter',
  templateUrl: './interpreter.component.html',
  styleUrls: ['./interpreter.component.scss']
})
export class InterpreterComponent implements OnInit {
  // Add Modal
  addModalTitle: string;
  addModalVisible: boolean;
  newInterpreterName: string = '';
  newInterpreterType: string = '';

  constructor(private fb: FormBuilder) {
    this.interpreterQueryForm = this.fb.group({
      interpreterName: [''],
      interpreterStatus: ['']
    });
  }

  statusOptions: CascaderOption[] = [
    {
      value: 'Running',
      label: 'Running',
      isLeaf: true
    },
    {
      value: 'Idle',
      label: 'Idle',
      isLeaf: true
    }
  ];

  statusColor: {[key: string]: string} = {
    Running: 'blue',
    Idle: 'green'
  }
  interpreterQueryForm: FormGroup
  // TODO(kevin85421)
  lastUpdatedTime: number = Date.now();
  interpreterInfoList: InterpreterInfo[] = [
    {
      name: 'Spark Interpreter 1',
      type: 'Spark',
      status: 'Running',
      progress: 50,
      lastUpdated: this.lastUpdatedTime
    },
    {
      name: 'Python Interpreter 1',
      type: 'Python',
      status: 'Idle',
      progress: 65,
      lastUpdated: this.lastUpdatedTime
    }
  ]

  // TODO(kevin85421)
  queryInterpreter() {
    for (const key in this.interpreterQueryForm.controls) {
      console.log(key);
      console.log(this.interpreterQueryForm.controls[key].value);
    }
  }

  // TODO(kevin85421)
  killInterpreter() {}

  ngOnInit() {
  }

  onShowAddInterpreterModal() {
    this.addModalTitle = "Add";
    this.addModalVisible = true;
  }

  onHideAddInterpreterModal() {
    this.addModalVisible = false;
  }

  updateNewInterpreter(newInterpreter: {interpreterName: string, interpreterType: string}) {
    this.interpreterInfoList = [
      ...this.interpreterInfoList,
      {
        name: newInterpreter.interpreterName,
        type: newInterpreter.interpreterType,
        status: 'Idle',
        progress: 0,
        lastUpdated: this.lastUpdatedTime
      }
    ]
    this.onHideAddInterpreterModal();
  }
}
