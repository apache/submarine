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

import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Rest } from '@submarine/interfaces';
import { SysDeptSelect } from '@submarine/interfaces/sys-dept-select';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { BaseApiService } from './base-api.service';

@Injectable({
  providedIn: 'root'
})
export class DepartmentService {

  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {
  }

  fetchSysDeptSelect(): Observable<SysDeptSelect[]> {
    const apiUrl = this.baseApi.getRestApi('/sys/dept/queryIdTree');
    return this.httpClient.get<Rest<SysDeptSelect[]>>(apiUrl).pipe(
      switchMap(res => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }
}
