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

import { Component, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { NotebookInfo } from "@submarine/interfaces/notebook-interfaces/notebook-info";
import { NotebookService } from '@submarine/services/notebook-services/notebook.service';
import { UserService } from '@submarine/services/user.service';
import { isEqual } from 'lodash';
import { NzMessageService } from 'ng-zorro-antd/message';
import { interval, Subscription } from 'rxjs';
import { NotebookFormComponent } from './notebook-form/notebook-form.component';
import { TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'submarine-notebook-home',
  templateUrl: './notebook-home.component.html',
  styleUrls: ['./notebook-home.component.scss'],
})
export class NotebookHomeComponent implements OnInit, OnDestroy {
  // User Information
  userId;

  // Notebook list
  notebookList: NotebookInfo[] = [];

  subscribtions = new Subscription();

  @ViewChild('form', { static: true }) form: NotebookFormComponent;

  constructor(
    private notebookService: NotebookService,
    private nzMessageService: NzMessageService,
    private userService: UserService,
    private translate: TranslateService
  ) {}

  ngOnInit() {
    this.userService.fetchUserInfo().subscribe((res) => {
      this.userId = res.id;
      // get notebook list first
      this.refreshNotebook(true);
    });

    // add a loop to refresh notebook
    const resourceSub = interval(10000).subscribe(() => {
      this.refreshNotebook(false);
    });
    this.subscribtions.add(resourceSub);
  }

  /**
   * refresh notebook list
   * @param total total refresh
   */
  refreshNotebook(total: boolean) {
    this.notebookService.fetchNotebookList(this.userId).subscribe((res) => {
      if (total) {
        // Direct override of all, suitable in case of add/delete
        this.notebookList = res;
      } else {// Partial refresh required
        // exists list size
        const currentListSize = this.notebookList.length;
        // The backend returns a real-time list
        const newListSize = res.length;
        for (let i = 0; i < newListSize; i++) {
          // The latest notebook info
          const notebook = res[i]
          // If a new row is found, insert it directly into
          if (i > currentListSize - 1) {
            this.notebookList = [...this.notebookList, notebook]
          } else {
            // Otherwise compare relevant information and update
            let current = this.notebookList[i];
            // compare
            const keys = Object.keys(current);
            for (const key of keys) {
              if (!isEqual(current[key], notebook[key])) {
                current[key] = notebook[key]
              }
            }
          }
        }
        // Delete redundant rows
        if (currentListSize > newListSize) {
          this.notebookList = this.notebookList.splice(0, newListSize - currentListSize);
        }
      }
    });
  }

  ngOnDestroy() {
    this.subscribtions.unsubscribe();
  }

  onDeleteNotebook(id: string) {
    this.notebookService.deleteNotebook(id).subscribe({
      next: (result) => {},
      error: (msg) => {
        this.nzMessageService.error(`${msg}, ` + this.translate.instant('please try again'), {
          nzPauseOnHover: true,
        });
      },
      complete: () => {
        this.nzMessageService.info(this.translate.instant(`Deleted Notebook!`), {
          nzPauseOnHover: true,
        });
        // refresh notebook list after deleted a row
        this.refreshNotebook(true);
      },
    });
  }
}
