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

import { Component, OnInit, ViewContainerRef } from '@angular/core';
import { FormArray, FormControl, FormGroup, Validators } from '@angular/forms';
import { ActivatedRoute, NavigationStart, Router } from '@angular/router';
import { ExperimentInfo } from '@submarine/interfaces/experiment-info';
import { ExperimentService } from '@submarine/services/experiment.service';
import { nanoid } from 'nanoid';
import { NzMessageService, NzModalService } from 'ng-zorro-antd';
import { ExperimentSpec } from '@submarine/interfaces/experiment-spec';
import { ExperimentCustomizedForm } from './experiment-customized-form/experiment-customized-form.component';

@Component({
  selector: 'submarine-experiment',
  templateUrl: './experiment.component.html',
  styleUrls: ['./experiment.component.scss']
})
export class ExperimentComponent implements OnInit {
  experimentList: ExperimentInfo[] = [];
  checkedList: boolean[] = [];
  selectAllChecked: boolean = false;

  // Modal instance
  modal = null;

  // About experiment information
  isInfo = false;
  experimentID: string;

  // About show existing experiments
  showExperiment = 'All';
  searchText = '';

  // About new experiment
  okText = 'Next step';
  isVisible = false;
  formType = null;

  // About form management
  current = 0;

  // About update and clone
  mode: 'create' | 'update' | 'clone' = 'create';
  targetId: string = null;
  targetSpec: ExperimentSpec = null;

  statusColor: { [key: string]: string } = {
    Accepted: 'gold',
    Created: 'white',
    Running: 'green',
    Succeeded: 'blue'
  };

  constructor(
    private nzMessageService: NzMessageService,
    private route: ActivatedRoute,
    private router: Router,
    private modalService: NzModalService,
    private viewContainerRef: ViewContainerRef,
    private experimentService: ExperimentService
  ) {}

  ngOnInit() {
    this.fetchExperimentList();
    this.isInfo = this.router.url !== '/workbench/experiment';
    this.experimentID = this.route.snapshot.params.id;
    this.router.events.subscribe((val) => {
      if (val instanceof NavigationStart) {
        console.log(val.url);
        if (val.url === '/workbench/experiment') {
          this.isInfo = false;
          this.fetchExperimentList();
        } else {
          this.isInfo = true;
        }
      }
    });

    this.reloadCheck();
  }

  createModal(formType: 'customized' | 'pre') {
    this.modal = this.modalService.create({
      nzTitle: 'Create experiment',
      nzContent: ExperimentCustomizedForm,
    })
  }

  fetchExperimentList() {
    this.experimentService.fetchExperimentList().subscribe((list) => {
      this.experimentList = list;
      const currentTime = new Date();
      this.experimentList.forEach((item) => {
        if (item.status === 'Succeeded') {
          const finTime = new Date(item.finishedTime);
          const runTime = new Date(item.runningTime);
          const result = (finTime.getTime() - runTime.getTime()) / 1000;
          item.duration = this.experimentService.durationHandle(result);
        } else if (item.runningTime) {
          const runTime = new Date(item.runningTime);
          const result = (currentTime.getTime() - runTime.getTime()) / 1000;
          item.duration = this.experimentService.durationHandle(result);
        }
      });
      this.checkedList = [];
      for (let i = 0; i < this.experimentList.length; i++) {
        this.checkedList.push(false);
      }
    });
  }

  onDeleteExperiment(id: string, onMessage: boolean) {
    this.experimentService.deleteExperiment(id).subscribe(
      () => {
        if (onMessage === true) {
          this.nzMessageService.success('Delete Experiment Successfully!');
        }
        this.experimentService.fetchExperimentList();
      },
      (err) => {
        if (onMessage === true) {
          this.nzMessageService.success(err.message);
        }
      }
    );
  }

  reloadCheck() {
    /*
      When reload in info page, ths experimentId will turn into undefined, it will cause breadcrumb miss experimentId.
      Location.pathname -> /workbench/experiment/info/{experimentID}
      So slice out experimentId string from location.pathname to reassign experimentId.
      */
    if (location.pathname !== '/workbench/experiment') {
      const sliceString = String('/workbench/experiment/info');
      this.experimentID = location.pathname.slice(sliceString.length + 1);
    }
  }

  deleteExperiments() {
    for (let i = this.checkedList.length - 1; i >= 0; i--) {
      if (this.checkedList[i] === true) {
        this.onDeleteExperiment(this.experimentList[i].experimentId, false);
      }
    }

    this.selectAllChecked = false;
  }

  selectAll() {
    for (let i = 0; i < this.checkedList.length; i++) {
      this.checkedList[i] = this.selectAllChecked;
    }
  }

  // TODO(jasoonn): Filter experiment list
  filter(event) {
    console.log(this.searchText + event.key);
  }
  // TODO(jasoonn): Perform part of list
  showChange() {
    console.log('Change to ' + this.showExperiment);
  }
  // TODO(jasoonn): Start experiment
  startExperiment(Experiment) {
    console.log(Experiment);
  }
}
