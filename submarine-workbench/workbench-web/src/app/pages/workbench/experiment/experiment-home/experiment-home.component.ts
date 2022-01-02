/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { Component, OnInit, ViewChild } from '@angular/core';
import { ExperimentInfo } from '@submarine/interfaces/experiment-info';
import { ExperimentFormService } from '@submarine/services/experiment.form.service';
import { ExperimentService } from '@submarine/services/experiment.service';
import { NzMessageService } from 'ng-zorro-antd';
import { interval } from 'rxjs';
import { filter, mergeMap, take, tap, timeout, retryWhen } from 'rxjs/operators';
import { ExperimentFormComponent } from './experiment-form/experiment-form.component';

@Component({
  selector: 'app-experiment-home',
  templateUrl: './experiment-home.component.html',
  styleUrls: ['./experiment-home.component.scss'],
})
export class ExperimentHomeComponent implements OnInit {
  /*
  experiment-list property:
    p.s. CheckedList does not need eventListener to update,
    because when we modify the array in child component,
    the modification will be sync to parent.
  */
  experimentList: ExperimentInfo[];
  isListLoading: boolean = true;
  checkedList: boolean[];
  selectAllChecked: boolean = false;
  switchValue: boolean = true;

  // auto reload
  reloadPeriod: number = 3000; // default 3s
  reloadInterval = interval(this.reloadPeriod);
  reloadSub = null;

  //mlflow
  isMlflowLoading: boolean = true;
  mlflowUrl: string = '';

  // tensorboard
  isTensorboardLoading: boolean = true;
  tensorboardUrl: string = '';

  @ViewChild('form', { static: true }) form: ExperimentFormComponent;

  constructor(
    private experimentService: ExperimentService,
    private experimentFormService: ExperimentFormService,
    private nzMessageService: NzMessageService
  ) {}

  ngOnInit() {
    this.experimentFormService.fetchListService.subscribe(() => {
      this.fetchExperimentList(false);
    });

    this.experimentService.emitInfo(null);
    this.getTensorboardInfo(1000, 50000);
    this.onSwitchAutoReload();
  }

  fetchExperimentList(isAutoReload: boolean) {
    this.experimentService.fetchExperimentList().subscribe(
      (list) => {
        this.isListLoading = false;
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
        if(!isAutoReload){
          // If it is auto-reloading, we do not want to change the state of checkbox.
          this.checkedList = [];
          for (let i = 0; i < this.experimentList.length; i++) {
            this.checkedList.push(false);
          }
        }
      },
      (error) => {
        console.error(error);
      }
    );
  }

  onDeleteExperiment(id: string, onMessage: boolean) {
    this.experimentService.deleteExperiment(id).subscribe(
      () => {
        if (onMessage === true) {
          this.nzMessageService.success('Delete Experiment Successfully!');
        }
        this.fetchExperimentList(true);
      },
      (err) => {
        if (onMessage === true) {
          this.nzMessageService.error(err.message);
        }
      }
    );
  }

  deleteExperiments() {
    for (let i = this.checkedList.length - 1; i >= 0; i--) {
      if (this.checkedList[i] === true) {
        this.onDeleteExperiment(this.experimentList[i].experimentId, false);
      }
    }
    this.selectAllChecked = false;
  }

  onInitModal(obj) {
    this.form.initModal(obj.initMode, obj.initFormType, obj.id, obj.spec);
  }

  onSwitchAutoReload() {
    if (this.switchValue) {
      this.reloadSub = this.reloadInterval.subscribe((res) => {
        this.fetchExperimentList(true);
      });
    } else {
      if (this.reloadSub) {
        this.reloadSub.unsubscribe();
      }
    }
  }

  getTensorboardInfo(period: number, due: number) {
    /*
      It will keep polling every ${period} msec, and stop polling whenever
        1. The tensorboard status turns from unavailble to available
        2. It takes over ${due} msec
    */

    interval(period)
      .pipe(
        mergeMap(() => this.experimentService.getTensorboardInfo()), // map interval observable to tensorboardInfo observable
        retryWhen((error) => error), //  retry to get tensorboardInfo
        tap((x) => console.log(x)), // monitoring the process
        filter((res) => res.available), // only emit the success ones
        take(1), // if succeed, stop emitting new value from source observable
        timeout(due) // if timeout, it will throw an error
      )
      .subscribe(
        (res) => {
          this.isTensorboardLoading = !res.available;
          this.tensorboardUrl = res.url;
        },
        (err) => console.log(err)
      );
  }
}
