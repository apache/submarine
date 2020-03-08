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
import { FormGroup, FormControl, Validators, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'submarine-job',
  templateUrl: './job.component.html',
  styleUrls: ['./job.component.scss']
})
export class JobComponent implements OnInit {

  //About show existing jobs
  showJob = 'All';
  searchText = '';
  joblist=[
    {
      name: 'Spark actuator',
      id: 1,
      owner: 'Frank',
      actuator: 'Spark Actuator',
      status: 'Running',
      progress: 85,
      lastRun: '2009-09-24 20:38:24'
    }
  ]
  //About new Job
  createJob: FormGroup;
  current = 0;
  okText = 'Next Step';
  isVisible = false;

  monitorObjects = ['Table1', 'Table2'];
  ruleTemplates = ['Template1', 'Template2'];
  ruleTypes = ['Strong', 'Weak'];

  scheduleCycles = ['Month', 'Week'];

  constructor() { }

  ngOnInit() {
    this.createJob =  new FormGroup({
      'jobName': new FormControl(null, Validators.required),
      'description': new FormControl(null, [Validators.required]),
      'monitorObject': new FormControl('Table1'),
      'ruleTemplate': new FormControl('Template1'),
      'ruleType': new FormControl('Strong'),
      'startDate': new FormControl(new Date()),
      'scheduleCycle': new FormControl('Month')
    });
  }

  handleOk(){
    if (this.current === 1){
      this.okText = 'Complete';
      this.current++;
    }
    else if (this.current === 2){
      this.okText = 'Next Step';
      this.current = 0;
      this.isVisible = false;
      //TODO(jasoonn): Create Real Job
      console.log(this.createJob);
    }
    else {
      this.current++;
    }
  }

  //TODO(jasoonn): Filter Job list
  filter(event){
    console.log(this.searchText+event.key);
  }
  //TODO(jasoonn): Perfrom part of list
  showChange(){
    console.log("Change to " + this.showJob);
  }
  //TODO(jasoonn): Start Job
  startJob(job){
    console.log(job);
  }
  //TODO(jasoonn): Edit job
  editJob(job){
    console.log(job);
  }
}
