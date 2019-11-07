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

import { Location, LocationStrategy, PathLocationStrategy } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, NavigationEnd, Router } from '@angular/router';
import _ from 'lodash';

interface HeaderInfo {
  title: string;
  description: string;
  breadCrumb: string[];
}

@Component({
  selector: 'submarine-manager',
  templateUrl: './manager.component.html',
  styleUrls: ['./manager.component.scss'],
  providers: [Location, { provide: LocationStrategy, useClass: PathLocationStrategy }]
})
export class ManagerComponent implements OnInit {
  private headerInfo: { [key: string]: HeaderInfo } = {
    user: {
      title: 'user',
      description: 'You can check the user, delete the user, lock and unlock the user, etc.',
      breadCrumb: ['manager', 'user']
    }
  };
  currentHeaderInfo: HeaderInfo;

  constructor(private route: ActivatedRoute, private location: Location, private router: Router) {
    this.router.events.subscribe(event => {
      if (event instanceof NavigationEnd) {
        const lastMatch = _.last(event.urlAfterRedirects.split('/'));
        this.currentHeaderInfo = this.headerInfo[lastMatch];
      }
    });
  }

  ngOnInit() {}
}
