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

import { Component, OnInit, ViewChild } from '@angular/core';
import { Subscription } from 'rxjs';
import { ExponentialBackoff } from '@submarine/services/notebook-services/polling';
import { NotebookService } from '@submarine/services/notebook-services/notebook.service';
import { NzMessageService } from 'ng-zorro-antd/message';
import { UserService } from '@submarine/services/user.service';
import { isEqual } from "lodash";
import { NzNotificationService } from 'ng-zorro-antd/notification';
import { NotebookFormComponent } from './notebook-form/notebook-form.component';

@Component({
    selector: 'submarine-notebook-home',
    templateUrl: './notebook-home.component.html',
    styleUrls: ['./notebook-home.component.scss']
  })

export class NotebookHomeComponent implements OnInit {

  // User Information
  userId;

  // Notebook list
  notebookList;

  // Sync //
  // Subscription
  subscriptions = new Subscription();
  // Poller
  poller: ExponentialBackoff;

  @ViewChild('form', {static: true}) form: NotebookFormComponent;

constructor(
  private notebookService: NotebookService,
  private nzMessageService: NzMessageService,
  private userService: UserService,
){}

ngOnInit() {
  this.poller = new ExponentialBackoff({ interval: 1000, retries: 3 });
    const resourcesSub = this.poller.start().subscribe(() => {
      this.userService.fetchUserInfo().subscribe((res) => {
        this.userId = res.id;
        this.notebookService.fetchNotebookList(this.userId).subscribe(resources => {
          if (!isEqual(this.notebookList, resources)) {
            this.notebookList = resources;
            this.poller.reset();
          }
        });
      });
    });
    
    this.subscriptions.add(resourcesSub);
}

ngOnDestroy() {
  this.subscriptions.unsubscribe();
}

onDeleteNotebook(id: string) {
  this.notebookService.deleteNotebook(id).subscribe({
    next: (result) => {
    },
    error: (msg) => {
      this.nzMessageService.error(`${msg}, please try again`, {
        nzPauseOnHover: true
      });
    },
    complete: () => {
      this.nzMessageService.info(`Delete Notebook...`, {
        nzPauseOnHover: true
      });
    }
  });
}

}
