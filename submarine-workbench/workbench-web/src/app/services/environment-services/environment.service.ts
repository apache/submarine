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
import { EnvironmentInfo } from '@submarine/interfaces/environment-interfaces/environment-info';
import { EnvironmentSpec } from '@submarine/interfaces/environment-interfaces/environment-spec';
import { BaseApiService } from '@submarine/services/base-api.service';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class EnvironmentService {
  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  fetchEnvironmentList(): Observable<EnvironmentInfo[]> {
    const apiUrl = this.baseApi.getRestApi('/v1/environment');
    return this.httpClient.get<Rest<EnvironmentInfo[]>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  createEnvironment(spec: object): Observable<EnvironmentSpec> {
    const apiUrl = this.baseApi.getRestApi(`/v1/environment/`);
    return this.httpClient.post<Rest<EnvironmentSpec>>(apiUrl, spec).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'post', spec);
        }
      })
    );
  }

  deleteEnvironment(name: string): Observable<EnvironmentInfo> {
    const apiUrl = this.baseApi.getRestApi(`/v1/environment/${name}`);
    return this.httpClient.delete<Rest<EnvironmentInfo>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'delete', name);
        }
      })
    );
  }
}
