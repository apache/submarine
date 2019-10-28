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

import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ComponentsModule } from '@submarine/components/components.module';
import { NgZorroAntdModule } from 'ng-zorro-antd';

import { DataDictComponent } from './data-dict/data-dict.component';
import { ManagerRoutingModule } from './manager-routing.module';
import { ManagerComponent } from './manager.component';
import { UserDrawerComponent } from './user-drawer/user-drawer.component';
import { UserPasswordModalComponent } from './user-password-modal/user-password-modal.component';
import { UserComponent } from './user/user.component';

@NgModule({
  declarations: [UserComponent, ManagerComponent, DataDictComponent, UserPasswordModalComponent, UserDrawerComponent],
  imports: [CommonModule, ManagerRoutingModule, NgZorroAntdModule, ComponentsModule, FormsModule, ReactiveFormsModule]
})
export class ManagerModule {}
