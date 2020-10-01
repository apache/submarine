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

import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { UserInfo } from '@submarine/interfaces';
import { AuthService, UserService } from '@submarine/services';
import { NzNotificationService } from 'ng-zorro-antd';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

interface SidebarMenu {
  title: string;
  iconType: string;
  routerLink?: string;
  disabled?: boolean;
  children?: Array<{
    title: string;
    routerLink?: string;
    disabled: boolean;
  }>;
}

@Component({
  selector: 'submarine-workbench',
  templateUrl: './workbench.component.html',
  styleUrls: ['./workbench.component.scss']
})

// ['data', 'model', 'workspace', 'interpreter'];

export class WorkbenchComponent implements OnInit {
  isCollapsed: boolean = false;
  highlighted: boolean = true;
  menus: SidebarMenu[] = [
    {
      title: 'Home',
      iconType: 'home',
      routerLink: '/workbench/home'
    },
    {
      title: 'Workspace',
      iconType: 'desktop',
      routerLink: '/workbench/workspace',
      disabled: true
    },
    {
      title: 'Notebook',
      iconType: 'book',
      routerLink: '/workbench/notebook'
    },
    {
      title: 'Interpreter',
      iconType: 'api',
      routerLink: '/workbench/interpreter',
      disabled: true
    },
    {
      title: 'Experiment',
      iconType: 'cluster',
      routerLink: '/workbench/experiment',
      disabled: false
    },
    {
      title: 'Environment',
      iconType: 'codepen',
      routerLink: '/workbench/environment'
    },
    {
      title: 'Data',
      iconType: 'bar-chart',
      routerLink: '/workbench/data',
      disabled: true
    },
    {
      title: 'Model',
      iconType: 'experiment',
      routerLink: '/workbench/model',
      disabled: true
    },
    {
      title: 'Manager',
      iconType: 'setting',
      disabled: false,
      children: [
        {
          title: 'User',
          routerLink: '/workbench/manager/user',
          disabled: false
        },
        {
          title: 'Data dict',
          routerLink: '/workbench/manager/dataDict',
          disabled: false
        },
        {
          title: 'Department',
          routerLink: '/workbench/manager/department',
          disabled: false
        }
      ]
    },
    {
      title: 'Data',
      iconType: 'bar-chart',
      routerLink: '/workbench/data',
      disabled: true
    },
    {
      title: 'Model',
      iconType: 'experiment',
      routerLink: '/workbench/model',
      disabled: true
    },
    {
      title: 'Workspace',
      iconType: 'desktop',
      routerLink: '/workbench/workspace',
      disabled: true
    },
    {
      title: 'Interpreter',
      iconType: 'api',
      routerLink: '/workbench/interpreter',
      disabled: true
    }
  ];
  userInfo$: Observable<UserInfo>;

  constructor(
    private router: Router,
    private authService: AuthService,
    private userService: UserService,
    private nzNotificationService: NzNotificationService
  ) {}

  ngOnInit() {
    if (this.authService.isLoggedIn) {
      this.userInfo$ = this.userService.fetchUserInfo().pipe(
        tap((userInfo) => {
          this.nzNotificationService.success('Welcome', `Welcome back, ${userInfo.name}`);
        })
      );
    }
  }

  logout() {
    this.authService.logout().subscribe((isLogout) => {
      if (isLogout) {
        this.router.navigate(['/user/login']);
      }
    });
  }
}
