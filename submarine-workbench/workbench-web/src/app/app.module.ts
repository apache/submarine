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

import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { registerLocaleData } from '@angular/common';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import en from '@angular/common/locales/en';
import { FormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import {ApiTokenInjector} from "@submarine/core/api-token-injector";
import { LocalStorageService } from '@submarine/services';
import { en_US, NgZorroAntdModule, NZ_I18N } from 'ng-zorro-antd';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { IconsProviderModule } from './icons-provider.module';

registerLocaleData(en);

@NgModule({
  declarations: [AppComponent],
  imports: [
    BrowserModule,
    AppRoutingModule,
    IconsProviderModule,
    NgZorroAntdModule,
    FormsModule,
    HttpClientModule,
    BrowserAnimationsModule
  ],
  providers: [
    { provide: NZ_I18N, useValue: en_US },
    // add injector to set header a token when calling rest api
    { provide: HTTP_INTERCEPTORS, useClass: ApiTokenInjector, multi: true },
    LocalStorageService
  ],
  bootstrap: [AppComponent]
})
export class AppModule {}
