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
import { ModelInfo } from '@submarine/interfaces/model-info';
import { BaseApiService } from '@submarine/services/base-api.service';
import { of, throwError, Observable, Subject } from 'rxjs';
import { catchError, map, switchMap } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class ModelService {
  /*
    communicate between route-outlet and parent
    send experiment-id from ExperimentInfo to ExperimentHome
  */
  private emitInfoSource = new Subject<string>();
  infoEmitted$ = this.emitInfoSource.asObservable();

  constructor(private baseApi: BaseApiService, private httpClient: HttpClient) {}

  emitInfo(name: string) {
    this.emitInfoSource.next(name);
  }

  fetchModelList(): Observable<ModelInfo[]> {
    const apiUrl = this.baseApi.getRestApi('/v1/registered-model');
    return this.httpClient.get<Rest<ModelInfo[]>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  querySpecificModel(name: string): Observable<ModelInfo> {
    const apiUrl = this.baseApi.getRestApi(`/v1/registered-model/${name}`);
    return this.httpClient.get<Rest<ModelInfo>>(apiUrl).pipe(
      switchMap((res) => {
        if (res.success) {
          return of(res.result);
        } else {
          throw this.baseApi.createRequestError(res.message, res.code, apiUrl, 'get');
        }
      })
    );
  }

  durationHandle(secs: number) {
    const hr = Math.floor(secs / 3600);
    const min = Math.floor((secs - hr * 3600) / 60);
    const sec = Math.round(secs) - hr * 3600 - min * 60;
    let showHr;
    let showMin;
    let showSec;
    if (hr < 10) {
      showHr = '0' + hr;
    } else {
      showHr = hr.toString();
    }
    if (min < 10) {
      showMin = '0' + min;
    } else {
      showMin = min.toString();
    }
    if (sec < 10) {
      showSec = '0' + sec;
    } else {
      showSec = sec.toString();
    }
    return `${showHr}:${showMin}:${showSec}`;
  }
}
