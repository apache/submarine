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

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { of, throwError, Observable, Subject } from 'rxjs';
import { BaseApiService } from '@submarine/services/base-api.service';
import { Rest } from '@submarine/interfaces';
import { ModelVersionInfo } from '@submarine/interfaces/model-version-info';
import { switchMap } from 'rxjs/operators';

@Injectable({
    providedIn: 'root',
})

export class ModelVersionService {

  private emitInfoSource = new Subject<string>();
  infoEmitted$ = this.emitInfoSource.asObservable();

  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  emitInfo(id: string) {
      this.emitInfoSource.next(id);
  }

  querySpecificModel(name: string, version: string) : Observable<ModelVersionInfo> {
      const apiUrl = this.baseApi.getRestApi('/v1/model-version/' + name + '/' + version + '/');
      return this.httpClient.get<Rest<ModelVersionInfo>>(apiUrl).pipe(
          switchMap((res) => {
              if (res.success) {
                  return of(res.result);
              } else {
                  throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
              }
          })
      );
  }
}