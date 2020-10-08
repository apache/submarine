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
import { Environment } from '@submarine/interfaces/environment-info';
import { BaseApiService } from '@submarine/services/base-api.service';
import { of, Observable } from 'rxjs';
import { switchMap } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class EnvironmentService {
  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  fetchEnvironmentList(): Observable<Environment[]> {
    const apiUrl = this.baseApi.getRestApi('/v1/environment');
    return this.httpClient.get<Rest<Environment[]>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          //console.log(res.result);
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  // TODO(kobe860219): Query environment
  querySpecificEnvironment(id: string) {}

  createEnvironment(spec: object): Observable<Environment> {
    const apiUrl = this.baseApi.getRestApi(`/v1/environment/`);
    return this.httpClient.post<Rest<Environment>>(apiUrl, spec).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'post', spec);
        }
      })
    );
  }

  // TODO(kobe860219): Update an environment
  updateEnvironment(updateData) {}

  deleteEnvironment(name: string): Observable<Environment> {
    const apiUrl = this.baseApi.getRestApi(`/v1/environment/${name}`);
    return this.httpClient.delete<Rest<Environment>>(apiUrl).pipe(
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
