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

import { Component, OnInit, Input } from '@angular/core';
import { ExperimentService } from '../../../../../services/experiment.service';
import { NzMessageService } from 'ng-zorro-antd';

@Component({
  selector: 'submarine-outputs',
  templateUrl: './outputs.component.html',
  styleUrls: ['./outputs.component.scss']
})
export class OutputsComponent implements OnInit {
  @Input() experimentOutputID: string;
  podNameArr;
  podLogArr;
  logDetailArr;
  isShowing = false;

  constructor(private experimentService: ExperimentService, private nzMessageService: NzMessageService) {}

  ngOnInit() {
    this.getExoerimentLog();
  }

  getExoerimentLog() {
    this.experimentService.getExperimentLog(this.experimentOutputID).subscribe(
      (result) => {
        this.podNameArr = result.logContent.map((item) => Object.values(item)[0]);
        this.podLogArr = result.logContent.map((item) => Object.values(item)[1]);
      },
      (err) => {
        this.nzMessageService.error('Cannot load log of ' + this.experimentOutputID);
        console.log(err);
      }
    );
  }

  show(i: number) {
    this.isShowing = true;
    this.logDetailArr = this.podLogArr[i];
  }
}
